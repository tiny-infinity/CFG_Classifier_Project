import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import helper_funcs as hf
import testing_funcs as testf
import concurrent.futures
import time
import os

def k_fold_partition(tsv_file_path, k):
    print(f"Loading data from {tsv_file_path}...")
    df = hf.load_tsv_file(tsv_file_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    indices = np.array_split(np.arange(len(df)), k)
    list_of_dfs = [df.iloc[idx].reset_index(drop=True) for idx in indices]
    return list_of_dfs

def divide_datasets(list_of_dfs):

    k = len(list_of_dfs)
    ds_pairs = []

    for i in range(k):

        test_df = list_of_dfs[i].copy()
        train_dfs = [list_of_dfs[j] for j in range(k) if j != i]
        train_df = pd.concat(train_dfs).reset_index(drop=True)

        ds_pairs.append([train_df, test_df])

    return ds_pairs

def single_pair_test(ds_pair,tf_id,chr_id,fasta_file_path,markov_order):

    training_set = ds_pair[0]
    test_set = ds_pair[1]

    #TRAINING

    print(f"--- Processing Fold (Order {markov_order}) ---")

    bound_df = hf.stripped_df(df=training_set,
                              tf_id = tf_id,
                              bclass='B'
                              )
    
    unbound_df = hf.stripped_df(df=training_set,
                              tf_id = tf_id,
                              bclass='U'
                              )
    
    b_matrix = hf.construct_transition_matrix(markov_order=markov_order,
                                  fasta_file_path=f'{fasta_file_path}',
                                  target_df=bound_df,
                                  chr_id=f'{chr_id}',
                                  tf_id=f'{tf_id}')
    
    u_matrix = hf.construct_transition_matrix(markov_order=markov_order,
                                  fasta_file_path=f'{fasta_file_path}',
                                  target_df=unbound_df,
                                  chr_id=f'{chr_id}',
                                  tf_id=f'{tf_id}')
    
    #TESTING
    print("Beginning Testing...")

    test_res_df = hf.binding_prob_database(markov_order=markov_order,
                                           tf_data=test_set,
                                           fasta_file_path=fasta_file_path,
                                           chr_id=chr_id,
                                           bmatrix=b_matrix,
                                           umatrix=u_matrix)
    
    prs_vals = testf.prec_rec_spec(test_res_df=test_res_df,
                                  chr_id = chr_id,
                                  tf_id = tf_id,
                                  markov_order=markov_order)
    
    results = {'AU_PRC':testf.AU_PRC(prs_vals=prs_vals),
               'AU_ROC':testf.AU_ROC(prs_vals=prs_vals)}

    print(f"Result: {results}")
    return results

def single_pair_test(args):

    ds_pair,tf_id,chr_id,fasta_file_path,markov_order,fold_idx = args
    training_set = ds_pair[0]
    test_set = ds_pair[1]

    pid = os.getpid()
    print(f"[Process {pid}] Starting Fold {fold_idx + 1}...")

    start_time = time.time()

    #TRAINING

    print(f"--- Processing Fold (Order {markov_order}) ---")

    bound_df = hf.stripped_df(df=training_set,
                              tf_id = tf_id,
                              bclass='B'
                              )
    
    unbound_df = hf.stripped_df(df=training_set,
                              tf_id = tf_id,
                              bclass='U'
                              )
    
    b_matrix = hf.construct_transition_matrix(markov_order=markov_order,
                                  fasta_file_path=f'{fasta_file_path}',
                                  target_df=bound_df,
                                  chr_id=f'{chr_id}',
                                  tf_id=f'{tf_id}')
    
    u_matrix = hf.construct_transition_matrix(markov_order=markov_order,
                                  fasta_file_path=f'{fasta_file_path}',
                                  target_df=unbound_df,
                                  chr_id=f'{chr_id}',
                                  tf_id=f'{tf_id}')
    
    #TESTING
    print("Beginning Testing...")

    test_res_df = hf.binding_prob_database(markov_order=markov_order,
                                           tf_data=test_set,
                                           fasta_file_path=fasta_file_path,
                                           chr_id=chr_id,
                                           bmatrix=b_matrix,
                                           umatrix=u_matrix)
    
    prs_vals = testf.prec_rec_spec(test_res_df=test_res_df,
                                  chr_id = chr_id,
                                  tf_id = tf_id,
                                  markov_order=markov_order)
    
    results = {
        'Fold': fold_idx + 1,
        'AU_PRC': testf.AU_PRC(prs_vals=prs_vals),
        'AU_ROC': testf.AU_ROC(prs_vals=prs_vals),
        'Time_Sec': round(time.time() - start_time, 2)
    }



    print(f"[Process {pid}] Finished Fold {fold_idx + 1}. Time: {results['Time_Sec']}s")
    return results

def run_kfold_parallel(tsv_path, fasta_path, tf_id, chr_id, markov_order, k=10):
    
    # 1. Prepare partitions 
    partitions = k_fold_partition(tsv_path, k)
    ds_pairs = divide_datasets(partitions)
    
    # 2. Prepare arguments for parallel workers
    # We pack everything into a tuple so we can use map()
    tasks = []
    for i, pair in enumerate(ds_pairs):
        tasks.append((pair, tf_id, chr_id, fasta_path, markov_order, i))

    print(f"\n--- Starting Parallel Execution on {os.cpu_count()} Cores ---\n")
    
    results = []
    
    # 3. Parallel Execution
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # map returns results in the order they were started
        for res in executor.map(single_pair_test, tasks):
            results.append(res)

    # 4. Aggregate Results
    results_df = pd.DataFrame(results)
    print(f"\n--- Final K-Fold Results; m={markov_order},k={k}---")
    print(results_df)
    print("\nAverage AU_PRC:", results_df['AU_PRC'].mean(), "Standard Deviation: ", results_df['AU_PRC'].std())
    print("Average AU_ROC:", results_df['AU_ROC'].mean(), "Standard Deviation: ", results_df['AU_ROC'].std())
    
    return results_df


if __name__ == "__main__":
    # CONFIGURATION
    TSV_PATH = 'projectData/chr1_200bp_bins.tsv'
    FASTA_PATH = 'projectData/chr1.fa'
    TF_ID = 'REST'
    CHR_ID = 'chr1'
    MARKOV_ORDER = 5
    K_FOLDS = 10

    # Run safely
    try:
        final_results = run_kfold_parallel(tsv_path=TSV_PATH, 
                                           fasta_path=FASTA_PATH,
                                           tf_id=TF_ID, 
                                           chr_id=CHR_ID, 
                                           markov_order=MARKOV_ORDER, 
                                           k=K_FOLDS)
    except Exception as e:
        print(f"An error occurred: {e}")




    










    


