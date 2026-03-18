"""
FlowMIA-GAN: GAN-based membership inference attack for synthetic tabular data.

The discriminator is trained to distinguish real synthetic samples from
generator-produced noise. After training, its confidence scores are used
as a membership signal: samples that score high are more likely to be
training members.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Neural network modules
# ---------------------------------------------------------------------------


class Generator(nn.Module):
    """
    Fully connected generator with BatchNorm, LeakyReLU, and Dropout layers.

    Args:
        latent_dim: Dimension of the input noise vector.
        output_dim: Dimension of the generated sample (matches encoded data).
        hidden_dims: List of hidden layer widths.
    """

    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list):
        super().__init__()

        dims = [latent_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
            ]

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(z))


class Discriminator(nn.Module):
    """
    Fully connected discriminator with optional spectral normalization.

    Args:
        input_dim: Dimension of the input sample.
        hidden_dims: List of hidden layer widths.
        use_spectral_norm: Whether to apply spectral normalization (recommended
            for WGAN-GP training).
    """

    def __init__(self, input_dim: int, hidden_dims: list, use_spectral_norm: bool = True):
        super().__init__()

        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers += [linear, nn.LeakyReLU(0.2), nn.Dropout(0.3)]

        self.features = nn.Sequential(*layers)
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class FlowMIA_GAN:
    """
    GAN-based membership inference attack.

    Expects pre-transformed numpy arrays (numerical scaling + one-hot encoding
    + IP normalization already applied by the caller). This keeps preprocessing
    logic centralized in :class:`FlowMIA` and out of the attack itself.

    Args:
        X_member: Encoded training (member) samples, shape (N, D).
        X_non_member: Encoded held-out (non-member) samples, shape (M, D).
        X_synth: Encoded synthetic samples, shape (S, D).
        batch_size: Mini-batch size for GAN training.
        latent_dim: Noise vector dimension fed to the generator.
        generator_hidden: Hidden layer widths for the generator.
        discriminator_hidden: Hidden layer widths for the discriminator.
        lr_g: Learning rate for the generator optimizer.
        lr_d: Learning rate for the discriminator optimizer.
        use_wgan: If True, uses WGAN-GP loss; otherwise uses standard BCE.
        lambda_gp: Gradient penalty coefficient (WGAN-GP only).
        device: Torch device. Defaults to CUDA if available.
    """

    def __init__(
        self,
        X_member: np.ndarray,
        X_non_member: np.ndarray,
        X_synth: np.ndarray,
        batch_size: int = 128,
        latent_dim: int = 128,
        generator_hidden: list = [512, 512, 256],
        discriminator_hidden: list = [512, 256, 128],
        lr_g: float = 0.0001,
        lr_d: float = 0.0001,
        use_wgan: bool = True,
        lambda_gp: float = 10.0,
        device: torch.device = None,
    ):
        self.X_member = X_member
        self.X_non_member = X_non_member
        self.X_synth = X_synth

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.generator_hidden = generator_hidden
        self.discriminator_hidden = discriminator_hidden
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.use_wgan = use_wgan
        self.lambda_gp = lambda_gp

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = X_synth.shape[1]

        self.generator: Generator = None
        self.discriminator: Discriminator = None

    # ------------------------------------------------------------------
    # WGAN-GP helper
    # ------------------------------------------------------------------

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient penalty term for WGAN-GP.

        Args:
            real: Batch of real (synthetic) samples.
            fake: Batch of generator-produced samples.

        Returns:
            Scalar gradient penalty tensor.
        """
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device).expand_as(real)
        interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        d_out = self.discriminator(interpolates)
        grads = torch.autograd.grad(
            outputs=d_out,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_out),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return ((grads.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        epochs: int,
        fcheckpoint: int,
        save_path: str,
        n_critic: int = 5,
        label_smoothing: float = 0.9,
        noise_std: float = 0.05,
    ) -> dict:
        """
        Train the GAN on the synthetic dataset.

        The generator learns to mimic the synthetic distribution while the
        discriminator learns to score samples. After training, the discriminator
        scores are used as the MIA signal.

        Args:
            epochs: Total number of training epochs.
            fcheckpoint: Save a checkpoint every ``fcheckpoint`` epochs.
            save_path: Root directory for checkpoint files.
            n_critic: Discriminator update steps per generator step.
            label_smoothing: Soft label value for real samples (standard GAN).
            noise_std: Std of Gaussian noise added to real samples for stability.

        Returns:
            Training history dict with ``d_loss``, ``g_loss``, and (WGAN) ``gp``.
        """
        fcheckpoint = min(fcheckpoint, epochs)

        dataset = TensorDataset(torch.tensor(self.X_synth, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.generator = Generator(self.latent_dim, self.input_dim, self.generator_hidden).to(self.device)
        self.discriminator = Discriminator(self.input_dim, self.discriminator_hidden, use_spectral_norm=self.use_wgan).to(self.device)

        opt_g = optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.0, 0.9))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(0.0, 0.9))
        criterion = None if self.use_wgan else nn.BCELoss()

        history = {"d_loss": [], "g_loss": [], "gp": []}
        ckpt_dir = os.path.join(save_path, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        print("Starting GAN training...")
        epoch_bar = tqdm(range(epochs), desc="Training GAN", unit="epoch")

        for epoch in epoch_bar:
            d_losses, g_losses, gp_losses = [], [], []

            for (real_data,) in dataloader:
                bsz = real_data.size(0)
                real_data = real_data.to(self.device)

                # -- discriminator steps --
                for _ in range(n_critic):
                    opt_d.zero_grad()
                    noisy_real = real_data + torch.randn_like(real_data) * noise_std
                    z = torch.randn(bsz, self.latent_dim, device=self.device)
                    fake_data = self.generator(z).detach()

                    if self.use_wgan:
                        gp = self._gradient_penalty(real_data, fake_data)
                        loss_d = -self.discriminator(noisy_real).mean() + self.discriminator(fake_data).mean() + self.lambda_gp * gp
                        gp_losses.append(gp.item())
                    else:
                        real_labels = torch.full((bsz, 1), label_smoothing, device=self.device)
                        fake_labels = torch.zeros(bsz, 1, device=self.device)
                        loss_d = criterion(self.discriminator(noisy_real), real_labels) + criterion(self.discriminator(fake_data), fake_labels)

                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                    opt_d.step()

                # -- generator step --
                opt_g.zero_grad()
                z = torch.randn(bsz, self.latent_dim, device=self.device)
                fake_data = self.generator(z)

                if self.use_wgan:
                    loss_g = -self.discriminator(fake_data).mean()
                else:
                    loss_g = criterion(self.discriminator(fake_data), torch.ones(bsz, 1, device=self.device))

                loss_g.backward()
                opt_g.step()

                d_losses.append(loss_d.item())
                g_losses.append(loss_g.item())

            history["d_loss"].append(np.mean(d_losses))
            history["g_loss"].append(np.mean(g_losses))

            postfix = {"D": f"{history['d_loss'][-1]:.4f}", "G": f"{history['g_loss'][-1]:.4f}"}
            if self.use_wgan:
                history["gp"].append(np.mean(gp_losses))
                postfix["GP"] = f"{history['gp'][-1]:.4f}"
            epoch_bar.set_postfix(postfix)

            if (epoch + 1) % fcheckpoint == 0:
                ckpt = {
                    "generator_state_dict": self.generator.state_dict(),
                    "discriminator_state_dict": self.discriminator.state_dict(),
                    "latent_dim": self.latent_dim,
                    "input_dim": self.input_dim,
                    "generator_hidden": self.generator_hidden,
                    "discriminator_hidden": self.discriminator_hidden,
                    "use_wgan": self.use_wgan,
                }
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                torch.save(ckpt, ckpt_path)
                print(f"\n  Checkpoint saved → {ckpt_path}")

        return history

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, X: np.ndarray) -> np.ndarray:
        """
        Return discriminator confidence scores for a pre-transformed array.

        Args:
            X: Float32 array of shape (N, D).

        Returns:
            1-D array of scores in [0, 1].
        """
        if self.discriminator is None:
            raise RuntimeError("Model not fitted. Call fit() or load_model() first.")

        self.discriminator.eval()
        tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            scores = self.discriminator(tensor).cpu().numpy().flatten()
        return scores

    def _random_score(self, seed: int = 42) -> np.ndarray:
        """
        Score a random noise matrix to serve as a baseline reference.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            1-D array of discriminator scores for random noise.
        """
        rng = np.random.default_rng(seed)
        noise = rng.random(self.X_synth.shape).astype(np.float32)
        return self._score(noise)

    # ------------------------------------------------------------------
    # Membership inference
    # ------------------------------------------------------------------

    def membership_inference(
        self,
        threshold_method: str = "statistical",
        threshold_value: float = 0.5,
        alpha: float = 0.05,
        test_size: float = 1000
    ) -> dict:
        """
        Perform membership inference using discriminator scores.

        Scores are computed for members, non-members, synthetic samples, and
        random noise. Classification metrics are derived from a threshold
        chosen by ``threshold_method``.

        Args:
            threshold_method: Strategy for selecting the decision threshold.
                One of ``"statistical"`` (default), ``"median"``, ``"mean"``,
                ``"optimal"`` (Youden's J), or ``"custom"``.
            threshold_value: Used only when ``threshold_method="custom"``.
            alpha: Significance level for the ``"statistical"`` method
                (threshold at the (1 - alpha) percentile of non-member scores).

        Returns:
            Dictionary containing raw score arrays, classification metrics
            (AUC, accuracy, precision, recall, F1), confusion matrix counts,
            score statistics, and Wasserstein distances between groups.
        """
        
        rng = np.random.default_rng(42)
        idx_m = rng.choice(len(self.X_member), size=test_size, replace=False)
        idx_nm = rng.choice(len(self.X_non_member), size=test_size, replace=False)
        
        s_mem = self._score(self.X_member[idx_m])
        s_non = self._score(self.X_non_member[idx_nm])
        s_syn = self._score(self.X_synth)
        s_rnd = self._random_score()

        y_true = np.hstack([np.ones(len(s_mem)), np.zeros(len(s_non))])
        s_all = np.hstack([s_mem, s_non])

        # Threshold selection
        if threshold_method == "median":
            threshold = np.median(s_all)
        elif threshold_method == "mean":
            threshold = np.mean(s_all)
        elif threshold_method == "optimal":
            fpr, tpr, thresholds = roc_curve(y_true, s_all)
            threshold = thresholds[np.argmax(tpr - fpr)]
        elif threshold_method == "statistical":
            threshold = np.percentile(s_non, (1 - alpha) * 100)
        elif threshold_method == "custom":
            threshold = threshold_value
        else:
            raise ValueError(
                "threshold_method must be one of: 'median', 'mean', 'optimal', 'statistical', 'custom'."
            )

        y_pred = (s_all >= threshold).astype(int)
        auc = roc_auc_score(y_true, s_all)
        acc = accuracy_score(y_true, y_pred)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        from scipy.stats import wasserstein_distance

        return {
            # Raw scores
            "score_members": s_mem,
            "score_non_members": s_non,
            "score_synthetic": s_syn,
            "score_random": s_rnd,
            # Classification metrics
            "auc": auc,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "threshold": threshold,
            "threshold_method": threshold_method,
            # Confusion matrix
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            # Score statistics
            "mean_score_members": float(s_mem.mean()),
            "mean_score_non_members": float(s_non.mean()),
            "mean_score_synthetic": float(s_syn.mean()),
            "mean_score_random": float(s_rnd.mean()),
            "std_score_members": float(s_mem.std()),
            "std_score_non_members": float(s_non.std()),
            "std_score_synthetic": float(s_syn.std()),
            "std_score_random": float(s_rnd.std()),
            # Gap metrics
            "score_gap_member_non_member": float(s_mem.mean() - s_non.mean()),
            "score_gap_member_synthetic": float(s_mem.mean() - s_syn.mean()),
            "score_gap_member_random": float(s_mem.mean() - s_rnd.mean()),
            # Privacy risk
            "privacy_risk_score": float(s_syn.mean() / (s_mem.mean() + 1e-10)),
            "wasserstein_member_synthetic": float(wasserstein_distance(s_mem, s_syn)),
            "wasserstein_nonmember_synthetic": float(wasserstein_distance(s_non, s_syn)),
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def load_model(self, filepath: str) -> None:
        """
        Load a previously saved GAN checkpoint.

        Args:
            filepath: Path to the ``.pth`` checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)

        self.latent_dim = ckpt["latent_dim"]
        self.input_dim = ckpt["input_dim"]
        self.generator_hidden = ckpt["generator_hidden"]
        self.discriminator_hidden = ckpt["discriminator_hidden"]
        self.use_wgan = ckpt.get("use_wgan", False)

        self.generator = Generator(self.latent_dim, self.input_dim, self.generator_hidden).to(self.device)
        self.discriminator = Discriminator(self.input_dim, self.discriminator_hidden, use_spectral_norm=self.use_wgan).to(self.device)

        self.generator.load_state_dict(ckpt["generator_state_dict"])
        self.discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        self.generator.eval()
        self.discriminator.eval()

        print(f"Model loaded from {filepath}")
        print(f"  latent_dim={self.latent_dim}, input_dim={self.input_dim}, wgan={self.use_wgan}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_score_distributions(self, results: dict, colors: dict, figsize: tuple = (5, 5)):
        """Plot overlapping score histograms for members, non-members, and synthetic."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(results["score_members"], bins=50, alpha=0.6, label="Members", color=colors["members"], density=True)
        ax.hist(results["score_non_members"], bins=50, alpha=0.6, label="Non-members", color=colors["non_members"], density=True)
        ax.hist(results["score_synthetic"], bins=50, alpha=0.6, label="Synthetic", color=colors["synthetic"], density=True)
        ax.set_xlabel("Discriminator Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc="upper center")
        ax.grid(True, alpha=0.3)
        return fig

    def plot_roc_and_pr_curves(self, results: dict, figsize: tuple = (10, 5)):
        """Plot ROC and Precision-Recall curves for the MIA."""
        y_true = np.hstack([np.ones(len(results["score_members"])), np.zeros(len(results["score_non_members"]))])
        s_all = np.hstack([results["score_members"], results["score_non_members"]])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        fpr, tpr, _ = roc_curve(y_true, s_all)
        ax1.plot(fpr, tpr, color="#000000", lw=2, label=f"AUC = {results['auc']:.4f}")
        ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        precision, recall, _ = precision_recall_curve(y_true, s_all)
        ax2.plot(recall, precision, color="#3498db", lw=2)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self, results: dict, figsize: tuple = (6, 5)):
        """Plot the confusion matrix for the chosen threshold."""
        y_true = np.hstack([np.ones(len(results["score_members"])), np.zeros(len(results["score_non_members"]))])
        s_all = np.hstack([results["score_members"], results["score_non_members"]])
        y_pred = (s_all >= results["threshold"]).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-Member", "Member"],
                    yticklabels=["Non-Member", "Member"],
                    ax=ax, cbar_kws={"label": "Count"})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.7, f"({cm[i, j] / cm.sum() * 100:.1f}%)",
                        ha="center", va="center", fontsize=10, color="gray")
        plt.tight_layout()
        return fig

    def plot_all(self, results: dict, colors: dict, save_path: str) -> None:
        """
        Generate and save all diagnostic plots.

        Args:
            results: Output of :meth:`membership_inference`.
            colors: Dict mapping group names to hex color strings.
            save_path: Root directory; plots are written to ``save_path/plots/``.
        """
        plot_dir = os.path.join(save_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        plots = [
            ("distributions", self.plot_score_distributions(results, colors)),
            ("roc_pr", self.plot_roc_and_pr_curves(results)),
            ("confusion", self.plot_confusion_matrix(results)),
        ]

        for name, fig in plots:
            path = os.path.join(plot_dir, f"{name}.pdf")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path}")