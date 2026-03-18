import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def fit_scale(synth, numerical_cols, categorical_cols, scaler, onehot_encoders):
        """Fit scalers on synthetic + non-member data"""
        scale_data = pd.concat([
            synth, 
            # self.non_member,     
        ], ignore_index=True)
        
        X_num = scale_data[numerical_cols].values
        scaler.fit(X_num)
        
        for col in categorical_cols:
            if col not in scale_data.columns:
                continue
            
            onehot_encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            onehot_encoders[col].fit(scale_data[[col]])
            
def transform_scale(df: pd.DataFrame, numerical_cols, categorical_cols, scaler, onehot_encoders) -> np.ndarray:        
	encoded_parts = []        

	
	if len(numerical_cols) > 0:
		X_num = df[numerical_cols].values 
		X_num = scaler.transform(X_num)
		encoded_parts.append(X_num)
		
	for col in categorical_cols:
		if col not in df.columns:
			continue            
		X_cat = onehot_encoders[col].transform(df[[col]])
		encoded_parts.append(X_cat)
		
	X_encoded = np.hstack(encoded_parts).astype(np.float32)   	
	
	return X_encoded