from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import os

class Utility:
    def __init__(self, classifiers, train, test, synth, categorical_cols, numerical_cols, ip_cols, label_col):
        self.num_cols = numerical_cols
        self.cat_cols = categorical_cols + ip_cols
        self.label_col = label_col
        
        self.classifers = classifiers
        self.X_train = train[self.num_cols + self.cat_cols]
        self.y_train = train[label_col]
        
        self.X_synth = synth[self.num_cols + self.cat_cols]
        self.y_synth = synth[label_col]
        
        self.X_test = test[self.num_cols + self.cat_cols]
        self.y_test = test[label_col] 


    def rtr(self, clf):
        
        rtr_dict = {
            'Accuracy': 0,
            'Precision': 0,
            'Recall': 0,
            'F1-Score': 0
        }
        preprocessor = ColumnTransformer(transformers=[
                                    ("num", RobustScaler(), self.num_cols),
                                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols)
                                ]
                            )
        pipeline_rtr = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("clf", clf)
            ]
        )

        pipeline_rtr.fit(self.X_train, self.y_train)
        y_pred_rtr = pipeline_rtr.predict(self.X_test)
        rtr_dict['Accuracy'] = accuracy_score(self.y_test, y_pred_rtr)
        rtr_dict['Precision'] = precision_score(self.y_test, y_pred_rtr, average='macro')
        rtr_dict['Recall'] = recall_score(self.y_test, y_pred_rtr, average='macro')
        rtr_dict['F1-Score'] = f1_score(self.y_test, y_pred_rtr, average='macro')
        return rtr_dict
    
    def tstr(self, clf):
        
        tstr_dict = {
            'Accuracy': 0,
            'Precision': 0,
            'Recall': 0,
            'F1-Score': 0
        }
        preprocessor = ColumnTransformer(transformers=[
                                    ("num", RobustScaler(), self.num_cols),
                                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols)
                                ]
                            )
        pipeline_tstr = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("clf", clf)
            ]
        )

        pipeline_tstr.fit(self.X_synth, self.y_synth)
        y_pred_tstr = pipeline_tstr.predict(self.X_test)
        tstr_dict['Accuracy'] = accuracy_score(self.y_test, y_pred_tstr)
        tstr_dict['Precision'] = precision_score(self.y_test, y_pred_tstr, average='macro')
        tstr_dict['Recall'] = recall_score(self.y_test, y_pred_tstr, average='macro')
        tstr_dict['F1-Score'] = f1_score(self.y_test, y_pred_tstr, average='macro')
        
        return tstr_dict

    
    def evaluate(self):
        utility_dict = {}
        for clf in self.classifers:
            print('Utility for classifier:', clf.__class__.__name__)
            print('Running RTR evaluation...')
            rtr_results = self.rtr(clf)
            print('Running TSTR evaluation...')
            tstr_results = self.tstr(clf)
            utility_dict[clf.__class__.__name__] = {
                "RTR": rtr_results,
                "TSTR": tstr_results
            }
        return utility_dict
    
    
    def plot_utility(self, utility_dict, save_path):
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        classifiers = list(utility_dict.keys())
        x = np.arange(len(classifiers))

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        axes = axes.flatten()

        bar_width = 0.6

        for i, metric in enumerate(metrics):
            tstr_values = [utility_dict[clf]["TSTR"][metric] for clf in classifiers]
            rtr_values  = [utility_dict[clf]["RTR"][metric]  for clf in classifiers]

            ax = axes[i]

            # TSTR como barra
            ax.bar(x, tstr_values, width=bar_width, alpha=0.7, label="TSTR")

            # RTR como ponto
            ax.scatter(x, rtr_values, zorder=3, label="RTR")

            ax.set_title(metric)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            if i >= 2:
                ax.set_xticks(x)
                ax.set_xticklabels(classifiers, rotation=30, ha="right")

            if i == 0:
                ax.legend()

        plt.tight_layout()
        save_path = os.path.join(save_path, 'plots')
        os.makedirs(save_path, exist_ok=True)
        util_path = os.path.join(save_path, 'utility.pdf')
        plt.savefig(util_path)
        plt.close()
    

