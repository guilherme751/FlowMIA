import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler 


class TSTRUtility:
    def __init__(self, classifiers, cat_cols, num_cols):
        self.classifiers = classifiers if isinstance(classifiers, list) else [classifiers]
        self.results = {}
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.scaler =  RobustScaler()
        self.onehot_encoders = {}
        # self.ip_cols = ip_cols
        # self.train = train
        # self.test = test
        # self.synth = synth
        
    def _encode(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
       
        df = df.copy()
        
        encoded_parts = []
        # encoded_parts.append(df[self.ip_cols].values/(2**32-1))  
        
        
        if fit:
            encoded_parts.append(self.scaler.fit_transform(df[self.num_cols]))
        else:
            encoded_parts.append(self.scaler.transform(df[self.num_cols]))
        
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if fit:
                self.onehot_encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                X_cat = self.onehot_encoders[col].fit_transform(df[[col]])
            else:
                X_cat = self.onehot_encoders[col].transform(df[[col]])
            
            encoded_parts.append(X_cat)
        
        X_encoded = np.hstack(encoded_parts).astype(np.float32)
        
        return X_encoded
    
    def train_on_synthetic(self, X_syn, y_syn):
        X = self._encode(X_syn, fit=True)
        for clf in self.classifiers:
            clf.fit(X, y_syn)
    
    def test_on_real(self, X_real, y_real):
        X = self._encode(X_real, fit=False)
        for clf in self.classifiers:
            name = clf.__class__.__name__
            y_pred = clf.predict(X)
            
            self.results[name] = {
                'accuracy': accuracy_score(y_real, y_pred),
                'precision': precision_score(y_real, y_pred, zero_division=0),
                'recall': recall_score(y_real, y_pred, zero_division=0),
                'f1': f1_score(y_real, y_pred, zero_division=0)
            }
        
        return self.results