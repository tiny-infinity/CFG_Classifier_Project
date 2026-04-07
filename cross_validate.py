import helper_funcs as hf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, 
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

def run_cv(tf_id='EP300', order=5, features_to_use=None):
    train_chrs = [1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 7, 18, 19, 20, 21, 22]
    
    log_odds_col = f'log_odds{tf_id}' 

    if features_to_use is None:
        if tf_id == 'EP300':
            features_to_use = ['ATAC', log_odds_col, 'FIMO_GATA3', 'FIMO_FOXA1', 'FIMO_CTCF', 'FIMO_REST', 'PhastCons']
        else:
            features_to_use = ['ATAC', log_odds_col, f'FIMO_{tf_id}', 'PhastCons']

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    auprc_scores = []
    auroc_scores = []
    
    all_y_true = []
    all_probs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_chrs)):
        cv_train = [train_chrs[i] for i in train_idx]
        cv_val = [train_chrs[i] for i in val_idx]
        
        train_df = hf.build_feature_matrix(cv_train, tf_id, order)
        val_df = hf.build_feature_matrix(cv_val, tf_id, order)

        X_train = train_df[features_to_use].copy()
        if 'ATAC' in X_train.columns:
            X_train['ATAC'] = X_train['ATAC'].map({'B': 1, 'U': 0})
        y_train = train_df[tf_id].map({'B': 1, 'U': 0})

        X_val = val_df[features_to_use].copy()
        if 'ATAC' in X_val.columns:
            X_val['ATAC'] = X_val['ATAC'].map({'B': 1, 'U': 0})
        y_val = val_df[tf_id].map({'B': 1, 'U': 0})

        model = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_val)[:, 1]
        
        auprc_scores.append(average_precision_score(y_val, probs))
        auroc_scores.append(roc_auc_score(y_val, probs))
        
        all_y_true.extend(y_val)
        all_probs.extend(probs)
        
        print(f"Fold {fold+1} AU-PRC: {auprc_scores[-1]:.4f} | AU-ROC: {auroc_scores[-1]:.4f}")

    print(f"\n--- Final Results for {tf_id} ---")
    print(f"Mean AU-PRC: {np.mean(auprc_scores):.4f} +/- {np.std(auprc_scores, ddof=1):.4f}")
    print(f"Mean AU-ROC: {np.mean(auroc_scores):.4f} +/- {np.std(auroc_scores, ddof=1):.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    precision, recall, _ = precision_recall_curve(all_y_true, all_probs)
    ax1.plot(recall, precision, label=f'(AUC={np.mean(auprc_scores):.2f})')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.legend()

    fpr, tpr, _ = roc_curve(all_y_true, all_probs)
    ax2.plot(fpr, tpr, label=f'(AUC={np.mean(auroc_scores):.2f})')
    ax2.plot([0, 1], [0, 1], 'k--') # Diagonal chance line
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{tf_id}_cv_curves.png")
    plt.show()

if __name__ == "__main__":
    run_cv(tf_id='CTCF',features_to_use=['log_oddsCTCF'])
    run_cv(tf_id='REST',features_to_use=['log_oddsREST'])
    