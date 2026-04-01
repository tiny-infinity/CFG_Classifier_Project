import helper_funcs as hf
import pandas as pd
import numpy as np

def run_verification(test_chrom_id=22, tf_id='REST', order=1):
    print(f"--- Starting Verification for {tf_id} on Chromosome {test_chrom_id} ---")
    
    # 1. Test Global Matrix Building (on a subset of 2 chroms to save time)
    print("Step 1: Testing Global Transition Matrix aggregation...")
    subset = [21, 22]
    prob_B, prob_U = hf.build_global_transition_matrix(subset, tf_id, order)
    
    # Check if probabilities sum to 1 (Verification of your Normalization fix)
    first_key_prefix = list(prob_B.keys())[0].split('_')[0]
    total_prob = sum(prob_B[f"{first_key_prefix}_{base}"] for base in ['A', 'T', 'G', 'C'])
    print(f"Check: Sum of probabilities for prefix '{first_key_prefix}': {total_prob:.4f}")
    
    if not np.isclose(total_prob, 1.0):
        print("CRITICAL: Normalization failed. Check the position of your pseudocount logic.")
    else:
        print("SUCCESS: Normalization is correct.")

    # 2. Test Feature Matrix Assembly for one chromosome
    print("\nStep 2: Building feature matrix for Chromosome 22...")
    # This calls assign_global_log_odds, map_fimo_to_bins, and phastcons_to_bins
    test_df = hf.build_feature_matrix([test_chrom_id], tf_id, order)

    # 3. Validation Checks
    print("\n--- Data Integrity Report ---")
    print(f"Total Bins Processed: {len(test_df)}")
    
    # Check Pillar 1: Markov
    print(f"Markov Score Range: {test_df[f'log_odds{tf_id}'].min():.2f} to {test_df[f'log_odds{tf_id}'].max():.2f}")
    
    # Check Pillar 2: ATAC
    atac_counts = test_df['ATAC'].value_counts()
    print(f"ATAC Distribution: Open={atac_counts.get('B', 0)}, Closed={atac_counts.get('U', 0)}")

    # Check Pillar 3: FIMO
    fimo_hits = (test_df[f'FIMO_{tf_id}'] > 0).sum()
    print(f"Bins with FIMO Hits: {fimo_hits} ({ (fimo_hits/len(test_df))*100 :.2f}%)")
    
    # Check Pillar 4: PhastCons
    print(f"PhastCons Range: {test_df['PhastCons'].min():.4f} to {test_df['PhastCons'].max():.4f}")

    # 4. Check for NaNs (The project killer)
    if test_df.isnull().values.any():
        print("WARNING: NaNs detected in the final matrix! Check your .fillna(0.0) logic.")
    else:
        print("SUCCESS: No missing values detected.")

    print("\nVerification Complete. If the ranges and counts look sensible, proceed to full training.")

if __name__ == "__main__":
    run_verification()