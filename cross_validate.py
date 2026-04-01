import helper_funcs as hf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

def run_cross_validation(tf_id='REST', markov_order=1):
    # Use a representative subset for speed, or all 19 for final numbers
    cv_chroms = [1, 2, 4, 5] 
    results = []

    for test_chr in cv_chroms:
        train_set = [c for c in cv_chroms if c != test_chr]
        print(f"\n--- CV Fold: Validating on chr{test_chr} ---")

        train_df = hf.build_feature_matrix(train_set, tf_id, markov_order)
        val_df = hf.build_feature_matrix([test_chr], tf_id, markov_order)

        features = ['ATAC', f'log_odds{tf_id}', f'FIMO_{tf_id}', 'PhastCons']
        
        X_train = train_df[features].copy()
        X_train['ATAC'] = X_train['ATAC'].map({'B': 1, 'U': 0})
        y_train = train_df[tf_id].map({'B': 1, 'U': 0})

        X_val = val_df[features].copy()
        X_val['ATAC'] = X_val['ATAC'].map({'B': 1, 'U': 0})
        y_val = val_df[tf_id].map({'B': 1, 'U': 0})

        rf = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)

        probs = rf.predict_proba(X_val)[:, 1]
        auprc = average_precision_score(y_val, probs)
        results.append(auprc)
        print(f"chr{test_chr} AU-PRC: {auprc:.4f}")

    print(f"\nAverage Cross-Validation AU-PRC: {sum(results)/len(results):.4f}")

if __name__ == "__main__":
    run_cross_validation()