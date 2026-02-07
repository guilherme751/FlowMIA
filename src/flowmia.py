from src.privacy.flowmia_gan import FlowMIA_GAN
import pandas as pd
from sdmetrics.single_table import DCROverfittingProtection
from src.fidelity.fidelity import fidelity_compute
from src.utility.utility import Utility
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier



config = {
    'member_path': 'datasets/real/cidds_train.csv',
    'non_member_path': 'datasets/reference/ton.csv',
    'synth_path': 'datasets/synthetic/netshare.csv',
    'categorical_cols': ['proto', 'label'],
    'numerical_cols': ['srcport', 'dstport', 'td', 'pkt', 'byt'],
    'ip_cols': ['srcip', 'dstip'],
    'batch_size': 128,
    'num_epochs': 10,
    'fcheckpoint': 10,
    'save_path': 'results/teste1'    
}

class FlowMIA():
    def __init__(self, config):
        self.config = config
        self.member = pd.read_csv(self.config['member_path'])
        self.non_member = pd.read_csv(self.config['non_member_path'])
        self.synth = pd.read_csv(self.configconfig['synth_path'])
        
        self.flowmia_gan = FlowMIA_GAN( self.member, 
                                        self.non_member, 
                                        self.synth, 
                                        categorical_cols=self.config['categorical_cols'],
                                        numerical_cols=self.config['numerical_cols'],
                                        ip_cols=self.config['ip_cols'], 
                                        batch_size = self.config['batch_size'])
        self.colors = {
            'members': '#e74c3c',
            'non_members': '#3498db',
            'synthetic': '#2ecc71',
            'random': '#95a5a6'
        }
    def _build_metadata_dict_for_dcr(user_config: dict):
        # Template base (imutável)
        base_columns = {
            "srcip": {"sdtype": "numerical"},
            "srcport": {"sdtype": "numerical"},
            "dstip": {"sdtype": "numerical"},
            "dstport": {"sdtype": "numerical"},
            "proto": {"sdtype": "categorical"},
            "td": {"sdtype": "numerical"},
            "byt": {"sdtype": "numerical"},
            "pkt": {"sdtype": "numerical"},
            "label": {"sdtype": "categorical"}
        }

        # Copia para não modificar o template original
        columns = {k: v.copy() for k, v in base_columns.items()}

        categorical_cols = user_config.get("categorical_cols", [])
        numerical_cols = user_config.get("numerical_cols", [])
        ip_cols = user_config.get("ip_cols", [])

        # IPs SEMPRE numéricos (regra forte)
        for col in ip_cols:
            if col in columns:
                columns[col]["sdtype"] = "numerical"

        # Colunas categóricas definidas pelo usuário
        for col in categorical_cols:
            if col in columns and col not in ip_cols:
                columns[col]["sdtype"] = "categorical"

        # Colunas numéricas definidas pelo usuário
        for col in numerical_cols:
            if col in columns:
                columns[col]["sdtype"] = "numerical"

        metadata_dict = {
            "primary_key": None,
            "columns": columns
        }

        return metadata_dict
        
        
    def evaluate_privacy(self, plot=False, n_samples_dcr=15000):
        self.mia_training_history = self.flowmia_gan.fit(epochs=self.config['num_epochs'], 
                                       fcheckpoint=self.config['fcheckpoint'], 
                                       save_path=self.config['save_path'])
        
        self.mia_results = self.flowmia_gan.membership_inference()
        meta_dict = self._build_metadata_dict_for_dcr(self.config)
        DCROverfittingProtection.compute_breakdown( real_training_data=self.member.sample(n=n_samples_dcr, random_state=42), 
                                                    synthetic_data=self.synth.sample(n=n_samples_dcr, random_state=42),
                                                    real_validation_data=self.non_member.sample(n=n_samples_dcr, random_state=42),
                                                    metadata=meta_dict
                                                    )        
        
        if plot:
            self.flowmia_gan.plot_all(results=self.mia_results, colors=self.colors, save_path=self.config['save_path'])        
    
    
    def evaluate_fidelity(self):
        self.fidelity_results = fidelity_compute(self.member, self.synth, 
                         categorical=['srcip', 'srcport', 'dstip', 'dstport', 'proto','label'])
    
    def evaluate_utility(self):
        classifiers = [MLPClassifier(hidden_layer_sizes=(20,10), max_iter=50, random_state=42),
                    DecisionTreeClassifier(max_depth=12, min_samples_leaf=50),
                    KNeighborsClassifier(n_neighbors=5),
                    RandomForestClassifier(max_depth=12, min_samples_leaf=50, n_estimators=100, class_weight="balanced")]
        utility = Utility(classifiers, self.member, self.non_member, self.synth)
        utility_dict = utility.evaluate_utility()
        
        return utility_dict
        
        
        
        