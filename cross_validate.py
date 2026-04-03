import helper_funcs as hf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
import numpy as np

def run_cv(tf_id='EP300', order=5, features_to_use=None):
    train_chrs = [1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 7, 18, 19, 20, 21, 22]
    
    # Define the "Full" feature pool correctly
    # Note: Using f-string to match exactly how helper_funcs names the column
    log_odds_col = f'log_odds{tf_id}' 
    
    if features_to_use is None:
        if tf_id == 'EP300':
            features_to_use = ['ATAC', log_odds_col, 'FIMO_GATA3', 'FIMO_FOXA1', 'FIMO_CTCF', 'FIMO_REST', 'PhastCons']
        else:
            features_to_use = ['ATAC', log_odds_col, f'FIMO_{tf_id}', 'PhastCons']

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_chrs)):
        cv_train = [train_chrs[i] for i in train_idx]
        cv_val = [train_chrs[i] for i in val_idx]
        
        # Build matrices (this will now use your corrected helper_funcs names)
        train_df = hf.build_feature_matrix(cv_train, tf_id, order)
        val_df = hf.build_feature_matrix(cv_val, tf_id, order)

        # Ensure columns exist before subsetting to avoid the KeyError
        available_cols = train_df.columns.tolist()
        missing = [f for f in features_to_use if f not in available_cols]
        if missing:
            print(f"Error: Missing columns in DataFrame: {missing}")
            print(f"Available columns: {available_cols}")
            return

        X_train = train_df[features_to_use].copy()
        X_train['ATAC'] = X_train['ATAC'].map({'B': 1, 'U': 0})
        y_train = train_df[tf_id].map({'B': 1, 'U': 0})

        X_val = val_df[features_to_use].copy()
        X_val['ATAC'] = X_val['ATAC'].map({'B': 1, 'U': 0})
        y_val = val_df[tf_id].map({'B': 1, 'U': 0})

        model = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, class_weight='balanced')
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_val)[:, 1]
        auprc = average_precision_score(y_val, probs)
        results.append(auprc)
        print(f"Fold {fold+1} AU-PRC: {auprc:.4f}")

    print(f"\nMean CV AU-PRC: {np.mean(results):.4f}")

if __name__ == "__main__":
    # To test WITHOUT PhastCons or FIMO:
    # run_cv(tf_id='EP300', features_to_use=['ATAC', 'log_oddsEP300'])
    
    # To test FULL model:
    run_cv(tf_id='EP300',features_to_use=['ATAC', 'log_oddsEP300'])