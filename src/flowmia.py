import pandas as pd
from sdmetrics.single_table import DCROverfittingProtection
from src.privacy.flowmia_gan import FlowMIA_GAN
from src.fidelity.fidelity import fidelity_compute, plotFidelity
from src.utility.utility import Utility
import os

class FlowMIA():
    def __init__(self, config):
        self.config = config
        self.member = pd.read_csv(self.config['member_path'])
        self.non_member = pd.read_csv(self.config['non_member_path'])
        self.synth = pd.read_csv(self.config['synth_path'])
        self.util_test = pd.read_csv(self.config['test_path'])
        
        self.save_path = os.makedirs(self.config['save_path'], exist_ok=True)
        
        
        self.categorical_cols = self.config['categorical_cols']
        self.numerical_cols = self.config['numerical_cols']
        self.ip_cols = self.config['ip_cols']
        self.label_col = self.config['label_col']
        
        self.flowmia_gan = FlowMIA_GAN( self.member, 
                                        self.non_member, 
                                        self.synth, 
                                        categorical_cols=self.categorical_cols + [self.label_col],
                                        numerical_cols=self.numerical_cols,
                                        ip_cols=self.ip_cols, 
                                        batch_size = self.config['batch_size'])
        self.colors = {
            'members': '#e74c3c',
            'non_members': '#3498db',
            'synthetic': '#2ecc71',
            'random': '#95a5a6'
        }
    def _build_metadata_dict_for_dcr(self):
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

        columns = {k: v.copy() for k, v in base_columns.items()}

        categorical_cols = self.config.get("categorical_cols", [])
        numerical_cols = self.config.get("numerical_cols", [])
        ip_cols = self.config.get("ip_cols", [])

        for col in ip_cols:
            if col in columns:
                columns[col]["sdtype"] = "numerical"

        for col in categorical_cols:
            if col in columns and col not in ip_cols:
                columns[col]["sdtype"] = "categorical"

        for col in numerical_cols:
            if col in columns:
                columns[col]["sdtype"] = "numerical"

        metadata_dict = {
            "primary_key": None,
            "columns": columns
        }

        return metadata_dict

    def flowmiagan(self, plot = False):
        print('Starting FlowMIA privacy evaluation...')
        print('Training FlowMIA GAN...')
        self.mia_training_history = self.flowmia_gan.fit(epochs=self.config['num_epochs'], 
                                       fcheckpoint=self.config['fcheckpoint'], 
                                       save_path=self.config['save_path'])
        self.mia_results = self.flowmia_gan.membership_inference()
        print(f'FlowMIA GAN inference results: {self.mia_results}')
        if plot:
            self.flowmia_gan.plot_all(results=self.mia_results, colors=self.colors, save_path=self.config['save_path'])        
        return self.mia_training_history, self.mia_results
    def compute_dcr(self, n_sample=15000): 
        print('Starting DCR evaluation...')
        meta_dict = self._build_metadata_dict_for_dcr()
        self.dcr_results = DCROverfittingProtection.compute_breakdown( 
                        real_training_data= self.member.sample(n=n_sample, random_state=42), 
                        synthetic_data= self.synth.sample(n=n_sample, random_state=42),
                        real_validation_data= self.non_member.sample(n=n_sample, random_state=42),
                        metadata= meta_dict
                    ) 
        print(f'DCR evaluation results: {self.dcr_results}')   
        
    def evaluate_privacy(self, plot=False, dcr=False, n_samples_dcr=15000):
        self.flowmiagan(plot=plot)
        
        self.dcr_results = None
        if dcr:
            self.compute_dcr(n_sample=n_samples_dcr)
            
        return self.mia_results, self.dcr_results
    
    def evaluate_fidelity(self, plot=False):
        self.fidelity_results = fidelity_compute(self.member, self.synth, 
                                categorical=self.categorical_cols + [self.label_col])
        
        if plot:
            plotFidelity(self.fidelity_results, self.config['save_path'])
        return self.fidelity_results
    
    def evaluate_utility(self, classifiers: list, plot=False):
        
        utility = Utility(classifiers, self.member, self.util_test, self.synth, 
                          self.categorical_cols, self.numerical_cols, self.ip_cols, self.label_col)
        utility_dict = utility.evaluate()
        
        if plot:
            utility.plot_utility(utility_dict, self.config['save_path'])
        
        return utility_dict
        
        
        
        