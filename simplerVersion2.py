import helper_funcs as hf
import time
import argparse
import os
import pandas as pd

try:
    os.mkdir("./SimpleresultData")
    print("created result data directory")
except FileExistsError:
    print("Directory 'SimpleresultData' already exists.")
except FileNotFoundError:
    print("Parent directory does not exist.")
def single_pair_test_simpler(data_set, tf_id, chr_id, fasta_file_path, markov_order):
    """
    Trains and tests on the same data and returns the scored dataframe. 
    """
    
    # TRAINING
    print(f"--- Processing Order {markov_order} ---")

    bound_df = hf.stripped_df(df=data_set, tf_id=tf_id, bclass='B')
    unbound_df = hf.stripped_df(df=data_set, tf_id=tf_id, bclass='U')
    
    b_matrix = hf.construct_transition_matrix(markov_order=markov_order,
                                              fasta_file_path=fasta_file_path,
                                              target_df=bound_df,
                                              chr_id=chr_id,
                                              tf_id=tf_id)
    
    u_matrix = hf.construct_transition_matrix(markov_order=markov_order,
                                              fasta_file_path=fasta_file_path,
                                              target_df=unbound_df,
                                              chr_id=chr_id,
                                              tf_id=tf_id)
    
    # TESTING
    print("Beginning Testing...")

    test_res_df = hf.binding_prob_database(markov_order=markov_order,
                                           tf_data=data_set,
                                           fasta_file_path=fasta_file_path,
                                           chr_id=chr_id,
                                           bmatrix=b_matrix,
                                           umatrix=u_matrix)
    
    return test_res_df

def main():
    parser = argparse.ArgumentParser(description="Run Simpler Version with Markov model")

    parser.add_argument("--tf_id", required=True, help="Transcription factor ID (e.g., REST, EP300, CTCF)")
    parser.add_argument("--fasta_path", required=True, help="Relative file path for FASTA file")
    parser.add_argument("--markov_order", type=int, required=True, help="Markov order (0-10)")
    
    parser.add_argument("--tsv_path", required=True, help="Relative file path for TSV coordinate file")
    parser.add_argument("--chr_id", required=True, help="Chromosome ID (e.g., chr4)")

    args = parser.parse_args()
    start_time = time.time()
    
    try:
        print(f"Loading data from {args.tsv_path}...")
        data_set = hf.load_tsv_file(args.tsv_path)
        
        results_df = single_pair_test_simpler(data_set=data_set, 
                                              tf_id=args.tf_id, 
                                              chr_id=args.chr_id, 
                                              fasta_file_path=args.fasta_path, 
                                              markov_order=args.markov_order)
        
        print(f"\n--- Final Log-Likelihood Scores (Order {args.markov_order}) ---")
        score_col = f'Score_{args.markov_order}'
        
        print(results_df[['start', 'end', args.tf_id, score_col]]) 

        results_df.to_csv(f"SimpleresultData/simple_m{args.markov_order}{args.chr_id}.csv")
        print("Results saved in Simpleresultdata")

        
    except Exception as e:
        print(f"Error for markov_order={args.markov_order}: {e}")
        return
        
    end_time = time.time()
    print(f"\nTotal execution time: {round(end_time - start_time, 2)} seconds")

if __name__ == "__main__":
    
    main()