from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class Utility:
    def __init__(self, classifiers, train, test, synth):
        self.num_cols = ['srcport', 'dstport', 'td', 'pkt', 'byt']
        self.cat_cols = ['srcip', 'dstip', 'proto']        
        label_col = 'label'
        
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

    
    def evaluate_utility(self):
        utility_dict = {}
        for clf in self.classifers:
            rtr_results = self.rtr(clf)
            tstr_results = self.tstr(clf)
            utility_dict[clf.__class__.__name__] = {
                "RTR": rtr_results,
                "TSTR": tstr_results
            }
        return utility_dict
    
    
    def plot_utility(self, utility_dict):
        pass
    

