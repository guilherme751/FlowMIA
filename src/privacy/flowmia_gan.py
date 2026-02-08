import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from typing import Dict

import os        
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, accuracy_score, roc_curve
import seaborn as sns
from tqdm import tqdm




class Generator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list):
        super().__init__()
        
        layers = []
        dims = [latent_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.model(z)
    
    
class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(0.3))
            else:
                layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class FlowMIA_GAN:
    def __init__(
        self,
        member: pd.DataFrame,
        non_member: pd.DataFrame,
        synthetic: pd.DataFrame,        
        categorical_cols: list,
        numerical_cols: list,
        ip_cols: list,
        batch_size: int = 128,
        latent_dim: int = 128,
        generator_hidden: list = [256, 512, 256],
        discriminator_hidden: list = [256, 128, 64],
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        scaler_type: str = 'robust',
        device: str = None
    ):
        self.member = member
        self.non_member = non_member
        self.synthetic = synthetic
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.ip_cols = ip_cols
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.generator_hidden = generator_hidden
        self.discriminator_hidden = discriminator_hidden
        self.lr_g = lr_g
        self.lr_d = lr_d
        if scaler_type == 'robust':
            self.scaler = RobustScaler()  
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"scaler_type must be 'robust' or 'standard'")
        
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.generator = None
        self.discriminator = None
        self.onehot_encoders = {}
        self.encoded_dim = None
    
    def fit_scale(self):
        scale_data = pd.concat([
            self.synthetic, 
            self.non_member,     
        ], ignore_index=True)
        
        X_num = scale_data[self.numerical_cols].values
        self.scaler.fit(X_num)
        for col in self.categorical_cols:
            if col not in scale_data.columns:
                continue
            
            self.onehot_encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.onehot_encoders[col].fit(scale_data[[col]])
            
    def transform_scale(self, df: pd.DataFrame) -> np.ndarray:        
        encoded_parts = []        
        # process ips
        ips = df[self.ip_cols].values
        ips = ips/(2**32-1)
        encoded_parts.append(ips)
        
        if len(self.numerical_cols) > 0:
            X_num = df[self.numerical_cols].values 
            X_num = self.scaler.transform(X_num)
            encoded_parts.append(X_num)
            
        for col in self.categorical_cols:
            if col not in df.columns:
                continue            
            X_cat = self.onehot_encoders[col].transform(df[[col]])
            encoded_parts.append(X_cat)
            
        X_encoded = np.hstack(encoded_parts).astype(np.float32)    
        self.encoded_dim = X_encoded.shape
        
        return X_encoded
    
    

    def fit(
            self,    
            epochs,
            fcheckpoint,
            save_path,
            n_critic: int = 5,
            label_smoothing: float = 0.9,
            noise_std: float = 0.1,
            verbose: bool = False,
        ) -> Dict:        
            
        if fcheckpoint > epochs:
            fcheckpoint = epochs
            
        print('Fitting the pre processors...')
        self.fit_scale()
        processed_data = self.transform_scale(self.synthetic)
        dataset = TensorDataset(torch.tensor(processed_data, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.generator = Generator(
            self.latent_dim, 
            self.encoded_dim[1], 
            self.generator_hidden
        ).to(self.device)
        
        self.discriminator = Discriminator(
            self.encoded_dim[1],
            self.discriminator_hidden
        ).to(self.device)
        
        optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999))
        
        criterion = nn.BCELoss()
        
        history = {'d_loss': [], 'g_loss': []}
        checkpoints = []
        
        print('Preprocessors fitted. Starting GAN training...')
        save_path = os.path.join(save_path, 'checkpoints')
        os.makedirs(save_path, exist_ok=True)
        epoch_bar = tqdm(range(epochs), desc="Training GAN", unit="epoch")

        for epoch in epoch_bar:
            d_losses, g_losses = [], []
            
            for batch_idx, (real_data,) in enumerate(dataloader):
                batch_size = real_data.size(0)
                real_data = real_data.to(self.device)
                
                real_labels = torch.ones(batch_size, 1).to(self.device) * label_smoothing
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Treina o discriminador
                for _ in range(n_critic):
                    optimizer_d.zero_grad()
                    
                    noisy_real = real_data + torch.randn_like(real_data) * noise_std                    
                    d_real = self.discriminator(noisy_real)
                    loss_d_real = criterion(d_real, real_labels)
                    
                    z = torch.randn(batch_size, self.latent_dim).to(self.device)
                    fake_data = self.generator(z)
                    d_fake = self.discriminator(fake_data.detach())
                    loss_d_fake = criterion(d_fake, fake_labels)
                    
                    loss_d = loss_d_real + loss_d_fake
                    loss_d.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                    optimizer_d.step()
                
                # Treina o gerador
                optimizer_g.zero_grad()
                
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_data = self.generator(z)
                d_fake = self.discriminator(fake_data)
                loss_g = criterion(d_fake, real_labels)
                
                loss_g.backward()
                optimizer_g.step()
                
                d_losses.append(loss_d.item())
                g_losses.append(loss_g.item())
            
            # Histórico
            history['d_loss'].append(np.mean(d_losses))
            history['g_loss'].append(np.mean(g_losses))
            
            # Atualiza barra do tqdm
            epoch_bar.set_postfix({
                "D_loss": f"{history['d_loss'][-1]:.4f}",
                "G_loss": f"{history['g_loss'][-1]:.4f}"
            })
            
            # Checkpoint
            if (epoch + 1) % fcheckpoint == 0:
                checkpoint_dict = {
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'latent_dim': self.latent_dim,
                    'encoded_dim': self.encoded_dim,
                    'generator_hidden': self.generator_hidden,
                    'discriminator_hidden': self.discriminator_hidden,
                    'scaler': self.scaler,
                    'onehot_encoders': self.onehot_encoders,
                    'categorical_cols': self.categorical_cols,
                    'device': str(self.device)
                }
                
                checkpoint_path = os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save(checkpoint_dict, checkpoint_path)
                print(f"\n✓ Model checkpoint {epoch+1} saved to {checkpoint_path}")
                checkpoints.append((epoch+1, checkpoint_dict))
        
        return history

    
    def membership_score(self, query_data: pd.DataFrame) -> np.ndarray:        
        if self.discriminator is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.discriminator.eval()
        
        X_query = self.transform_scale(query_data)
        X_query = torch.FloatTensor(X_query).to(self.device)
        
        with torch.no_grad():
            scores = self.discriminator(X_query).cpu().numpy().flatten()
        
        return scores
    
    def random_score(self, seed = 42):
        np.random.seed(seed)
        random_noise = np.random.random(self.encoded_dim)
        random_noise = torch.FloatTensor(random_noise).to(self.device)
        self.discriminator.eval()
        
        with torch.no_grad():
            scores = self.discriminator(random_noise).cpu().detach().numpy().flatten()
        return scores
    
    
    def membership_inference(
        self,
        threshold_method: str = 'optimal',
        threshold_value: float = 0.5 
    ) -> Dict[str, float]:
               
        scores_members = self.membership_score(self.member)
        scores_non_members = self.membership_score(self.non_member)
        scores_random = self.random_score()
        scores_synthetic = self.membership_score(self.synthetic)
        
        # Ground truth
        y_true = np.hstack([
            np.ones(len(scores_members)),
            np.zeros(len(scores_non_members))
        ])
        
        scores_all = np.hstack([scores_members, scores_non_members])
        
        # Escolha de threshold
        if threshold_method == 'median':
            threshold = np.median(scores_all)
        elif threshold_method == 'mean':
            threshold = np.mean(scores_all)
        elif threshold_method == 'optimal':
            # Youden's J statistic: maximiza TPR - FPR
            fpr, tpr, thresholds = roc_curve(y_true, scores_all)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            threshold = thresholds[optimal_idx]
        elif threshold_method == 'custom':
            threshold = threshold_value
        else:
            raise ValueError("threshold_method must be 'median', 'mean', optimal' or custom")
        
        y_pred = (scores_all >= threshold).astype(int)
        
        auc = roc_auc_score(y_true, scores_all)
        acc = accuracy_score(y_true, y_pred)
        
        # Métricas adicionais
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return {
            'score_members': scores_members,
            'score_non_members': scores_non_members,
            'score_random': scores_random,
            'score_synthetic': scores_synthetic,
            'auc': auc,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'threshold': threshold,
            'threshold_method': threshold_method,
            'mean_score_members': scores_members.mean(),
            'mean_score_non_members': scores_non_members.mean(),
            'mean_score_random': scores_random.mean(),
            'mean_score_synthetic': scores_synthetic.mean(),
            'std_score_members': scores_members.std(),
            'std_score_non_members': scores_non_members.std(),
            'std_score_random': scores_random.std(),
            'std_score_synthetic': scores_synthetic.std(),
            'score_gap_member_non_member': scores_members.mean() - scores_non_members.mean(),
            'score_gap_member_random': scores_members.mean() - scores_random.mean(),
            'score_gap_member_synthetic': scores_members.mean() - scores_synthetic.mean(),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
        
        
    def load_model(self, filepath: str):
        """
        Carrega modelo completo.
        
        Args:
            filepath: Caminho do modelo salvo
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Restaura configurações
        self.latent_dim = checkpoint['latent_dim']
        self.encoded_dim = checkpoint['encoded_dim']
        self.generator_hidden = checkpoint['generator_hidden']
        self.discriminator_hidden = checkpoint['discriminator_hidden']
        self.categorical_cols = checkpoint['categorical_cols']
        
        # Restaura preprocessadores
        self.scaler = checkpoint['scaler']
        self.onehot_encoders = checkpoint['onehot_encoders']
        
        # Recria modelos com arquitetura correta
        self.generator = Generator(
            self.latent_dim,
            self.encoded_dim[1],
            self.generator_hidden
        ).to(self.device)
        
        self.discriminator = Discriminator(
            self.encoded_dim[1],
            self.discriminator_hidden
        ).to(self.device)
        
        # Carrega pesos
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Modo eval
        self.generator.eval()
        self.discriminator.eval()
        
        print(f"✓ Model loaded from {filepath}")
        print(f"  - Latent dim: {self.latent_dim}")
        print(f"  - Encoded dim: {self.encoded_dim}")
        print(f"  - Device: {self.device}")   
        
        
        
    def plot_score_distributions(self, results, colors, figsize=(5, 5)):
        """
        Plot 1: Distributions
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
        ax.hist(results['score_members'], bins=50, alpha=0.6, label='Membros', 
                color=colors['members'], density=True)
        ax.hist(results['score_non_members'], bins=50, alpha=0.6, label='Não-membros', 
                color=colors['non_members'], density=True)
        
        if results['score_synthetic'] is not None:
            ax.hist(results['score_synthetic'], bins=50, alpha=0.6, label='Sintético', 
                    color=colors['synthetic'], density=True)
        
        # if results['score_random'] is not None:
        #     ax.hist(results['score_random'], bins=50, alpha=0.6, label='Ruído Aleatório', 
        #             color=colors['random'], density=True)       
    
        
        ax.set_xlabel('Score do Discriminador')
        ax.set_ylabel('Densidade')
        ax.legend(fontsize=8, loc='upper center')
        ax.grid(True, alpha=0.3)
        
        return fig
        
        
    def plot_roc_and_pr_curves(self, results, figsize=(10, 5)):
        """
        Plot 3: Curvas ROC e Precision-Recall
        """
        y_true = np.hstack([
            np.ones(len(results['score_members'])),
            np.zeros(len(results['score_non_members']))
        ])
        scores_all = np.hstack([results['score_members'], results['score_non_members']])
        
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, scores_all)
        
        ax1 = ax[0]
        ax1.plot(fpr, tpr, color="#000000", linewidth=2, 
                label=f"AUC = {results.get('auc', 0):.4f}")
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatório')
        ax1.set_xlabel('Taxa de Falsos Positivos')
        ax1.set_ylabel('Taxa de Verdadeiros Positivos')
        # ax1.set_title('Curva ROC')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        #Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, scores_all)
        ax2 = ax[1]
        ax2.plot(recall, precision, color='#3498db', linewidth=2)
        ax2.set_xlabel('Recall (TPR)')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    
    def plot_confusion_matrix(self, results, figsize=(8, 6)):
        """
        Plot 5: Matriz de confusão
        """
        y_true = np.hstack([
            np.ones(len(results['score_members'])),
            np.zeros(len(results['score_non_members']))
        ])
        scores_all = np.hstack([results['score_members'], results['score_non_members']])
        
        threshold = results['threshold']        
        
        y_pred = (scores_all >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Não-Membro', 'Membro'],
                   yticklabels=['Não-Membro', 'Membro'],
                   ax=ax, cbar_kws={'label': 'Contagem'})
        
        ax.set_xlabel('Predito')
        ax.set_ylabel('Verdadeiro')
        # ax.set_title(f'Confusion Matrix (Threshold={threshold:.4f})')
        
        # Adiciona porcentagens
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / cm.sum() * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                       ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        return fig
    
    
    def plot_all(self, results, colors, save_path):
        """
        Gera todos os plots
        """
        save_path = os.path.join(save_path, 'plots')
        os.makedirs(save_path, exist_ok=True)
        y_true = np.hstack([
            np.ones(len(results['score_members'])),
            np.zeros(len(results['score_non_members']))
        ])
        scores_all = np.hstack([results['score_members'], results['score_non_members']])
        figs = []
        
        print("Generating Plot 1: Score Distributions...")
        figs.append(('distributions', self.plot_score_distributions(results, colors)))
        
        print("Generating Plot 2: ROC and PR Curves...")
        figs.append(('roc_pr', self.plot_roc_and_pr_curves(results))) 
        
        print("Generating Plot 3: Confusion Matrix...")
        figs.append(('confusion', self.plot_confusion_matrix(results)))
        
        
        for name, fig in figs:
            save = os.path.join(save_path, f"{name}.pdf")
            fig.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Saved: {save}")
            plt.close(fig)
        
        