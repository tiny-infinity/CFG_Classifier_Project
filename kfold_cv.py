import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import helper_funcs as hf
import testing_funcs as testf
import concurrent.futures
import time
import os

def k_fold_partition(tsv_file_path, k):
    """
    Partitions the dataset into k folds for cross-validation.
    Args:
        tsv_file_path (str) : Path to the TSV data file
        k (int) : Number of folds
    Output:
        list_of_dfs (list) : List of DataFrames for each fold"""
    print(f"Loading data from {tsv_file_path}...")
    df = hf.load_tsv_file(tsv_file_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    indices = np.array_split(np.arange(len(df)), k)
    list_of_dfs = [df.iloc[idx].reset_index(drop=True) for idx in indices]
    return list_of_dfs

def divide_datasets(list_of_dfs):
    """
    Divides the list of DataFrames into training and testing pairs for k-fold CV.
    Args:
        list_of_dfs (list) : List of DataFrames for each fold
    Output:
        ds_pairs (list) : List of [training_set, testing_set] pairs for each fold
    """

    k = len(list_of_dfs)
    ds_pairs = []

    for i in range(k):

        test_df = list_of_dfs[i].copy()
        train_dfs = [list_of_dfs[j] for j in range(k) if j != i]
        train_df = pd.concat(train_dfs).reset_index(drop=True)

        ds_pairs.append([train_df, test_df])

    return ds_pairs

def single_pair_test(args):
        
    """
    Tests a single train-test pair for given parameters.
    Also saves the result as a .csv file.
    Args:
        ds_pair (list) : [training_set, testing_set] DataFrames
        tf_id (str) : Transcription Factor ID
        chr_id (str) : Chromosome ID
        fasta_file_path (str) : Path to the FASTA file
        markov_order (int) : Markov Order
    Output:
        results (dict) : Dictionary containing AU_PRC and AU_ROC results
    """

    ds_pair,tf_id,chr_id,fasta_file_path,markov_order,k,fold_idx = args
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

    prs_vals.to_csv(f'resultData/m{markov_order}k{k}f{fold_idx + 1}.csv', index=False)
    
    results = {
        'Fold': fold_idx + 1,
        'AU_PRC': testf.AU_PRC(prs_vals=prs_vals),
        'AU_ROC': testf.AU_ROC(prs_vals=prs_vals),
        'Time_Sec': round(time.time() - start_time, 2)
    }



    print(f"[Process {pid}] Finished Fold {fold_idx + 1}. Time: {results['Time_Sec']}s")
    return results

def run_kfold_parallel(tsv_path, fasta_path, tf_id, chr_id, markov_order, k=10, num_cpus=None):

    """
    Runs k-fold cross-validation in parallel.
    Args:
        tsv_path (str) : Path to the TSV data file
        fasta_path (str) : Path to the FASTA file path
        tf_id (str) : Transcription Factor ID
        chr_id (str) : Chromosome ID
        markov_order (int) : Markov Order
        k (int) : Number of folds
        num_cpus(int) : Nuumber of cores to be utilized for parallel processing. Uses all available cores if set to None
    Output:
        results_df (DataFrame) : DataFrame containing results for each fold"""
    
    # 1. Prepare partitions 
    partitions = k_fold_partition(tsv_path, k)
    ds_pairs = divide_datasets(partitions)
    
    effective_cpus = num_cpus if num_cpus is not None else os.cpu_count()
    # 2. Prepare arguments for parallel workers
    # We pack everything into a tuple so we can use map()
    tasks = []
    for i, pair in enumerate(ds_pairs):
        tasks.append((pair, tf_id, chr_id, fasta_path, markov_order,k, i))

    print(f"\n--- Starting Parallel Execution on {effective_cpus} Cores ---\n")
    
    results = []
    
    # 3. Parallel Execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
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





    










    


