import helper_funcs as hf
import pandas as pd
from  sklearn.ensemble import RandomForestClassifier
import joblib

training_chroms = [1,2,4,5,6,8,9,11,12,13,14,15,16,17,18,19,20,21,22]

target_tf = 'EP300'

order = 5

print(f"Building Feature Matrix for {target_tf}...")
master_df = hf.build_feature_matrix(chrom_set=training_chroms,
                                    markov_order=order,
                                    tf_id=target_tf)


X = master_df[['ATAC', f'log_odds{target_tf}', f'FIMO_{target_tf}', 'PhastCons']].copy()
X['ATAC'] = X['ATAC'].map({'B': 1, 'U': 0}) #Label Encoding
Y = master_df[target_tf].map({'B': 1, 'U': 0})

print(f"Training Random Forest on {len(X)} bins...")

model = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42,class_weight='balanced')
model.fit(X, Y)

model_filename = f"{target_tf}_rf_model.joblib"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

importances = model.feature_importances_
for name, imp in zip(X.columns, importances):
    print(f"Feature: {name} | Importance: {imp:.4f}")
