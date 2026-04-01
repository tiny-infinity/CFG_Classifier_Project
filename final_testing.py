import helper_funcs as hf
import pandas as pd
import joblib
import gzip

def generate_predictions(tf_id, test_chroms=[3, 10, 17], markov_order=1):
    
    model_path = f"{tf_id}_rf_model.joblib"
    print(f"Loading model: {model_path}...")
    model = joblib.load(model_path)
    
    for chrom_num in test_chroms:
        chr_id = f"chr{chrom_num}"
        print(f"Processing {chr_id}...")
    
        test_df = hf.build_feature_matrix([chrom_num], tf_id, markov_order)
        
        features = ['ATAC', f'log_odds{tf_id}', f'FIMO_{tf_id}', 'PhastCons']
        X_test = test_df[features].copy()
        
        X_test['ATAC'] = X_test['ATAC'].map({'B': 1, 'U': 0})
        
        probs = model.predict_proba(X_test)[:, 1]
    
        submission_df = pd.DataFrame({
            'chrom': chr_id,
            'start': test_df['start'],
            'end': test_df['end'],
            'probability': probs
        })
        
        output_name = f"{tf_id}_{chr_id}_predictions.tsv.gz"
        submission_df.to_csv(output_name, sep='\t', index=False, compression='gzip')
        print(f"Saved: {output_name}")

if __name__ == "__main__":
    for tf in ['CTCF', 'REST', 'EP300']:
        try:
            generate_predictions(tf)
        except Exception as e:
            print(f"Could not generate for {tf}: {e}")