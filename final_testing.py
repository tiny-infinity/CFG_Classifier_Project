import helper_funcs as hf
import pandas as pd
import joblib

def generate_predictions(tf_id, Bmat, Umat, test_chroms=[3, 10, 17], markov_order=5):
    
    model_path = f"{tf_id}_rf_model.joblib"
    print(f"Loading model: {model_path}...")
    model = joblib.load(model_path)
    
    for chrom_num in test_chroms:
        chr_id = f"chr{chrom_num}"
        print(f"--- Processing {chr_id} ---")
    
        test_df = hf.build_feature_matrix([chrom_num], tf_id, markov_order, Bmatrix=Bmat, Umatrix=Umat)
        
        
        
        if tf_id == 'EP300':
            features = ['ATAC', f'log_odds{tf_id}','FIMO_GATA3','FIMO_FOXA1','FIMO_CTCF', 'FIMO_REST', 'PhastCons']
        else:
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
        print(f"SUCCESS: Saved {output_name}")

if __name__ == "__main__":
    target_tf = 'EP300'
    order = 5 

    try:
        print("Loading Transition Matrix")
        markov_data = joblib.load(f"{target_tf}_transition_matrices.joblib")
        Bmat = markov_data['Bmat']
        Umat = markov_data['Umat']
    except FileNotFoundError:
        print("Matrices not found. Building Matrix...")
        train_chrs = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]
        Bmat, Umat = hf.build_global_transition_matrix(train_chrs, target_tf, order)
        joblib.dump({'Bmat': Bmat, 'Umat': Umat}, f"{target_tf}_transition_matrices.joblib")

    generate_predictions(target_tf, Bmat, Umat, test_chroms=[3, 10, 17], markov_order=order)