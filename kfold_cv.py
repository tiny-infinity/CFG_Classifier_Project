import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import helper_funcs as hf
import testing_funcs as testf

def k_fold_partition(tsv_file_path, k):
    df = hf.load_tsv_file(tsv_file_path)
    # Use np.array_split to get indices, then use iloc to get DataFrames
    indices = np.array_split(np.arange(len(df)), k)
    list_of_dfs = [df.iloc[idx].reset_index(drop=True) for idx in indices]
    return list_of_dfs

def divide_datasets(list_of_dfs):

    k = len(list_of_dfs)
    ds_pairs = []

    for i in range(k):

        test_df = list_of_dfs[i]
        train_dfs = [list_of_dfs[j] for j in range(k) if j != i]
        train_df = pd.concat(train_dfs).reset_index(drop=True)

        ds_pairs.append([train_df, test_df])

    return ds_pairs

def single_pair_test(ds_pair,tf_id,chr_id,fasta_file_path,markov_order):

    training_set = ds_pair[0]
    test_set = ds_pair[1]

    #TRAINING

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

    test_res_df = hf.binding_prob_database(markov_order=markov_order,
                                           tf_data=test_set,fasta_file_path=fasta_file_path,
                                           chr_id=chr_id,
                                           bmatrix=b_matrix,
                                           umatrix=u_matrix)
    
    prs_vals = testf.prec_rec_spec(test_res_df=test_res_df,
                                  chr_id = chr_id,
                                  tf_id = tf_id,
                                  markov_order=markov_order)
    
    results = {'AU_PRC':testf.AU_PRC(prs_vals=prs_vals),
               'AU_ROC':testf.AU_ROC(prs_vals=prs_vals)}

    return results




    

    



sample_div = divide_datasets(k_fold_partition(tsv_file_path='projectData/chr1_200bp_bins.tsv',k=10))
print(f"Number of folds: {len(sample_div)}")
for i, (train_df, test_df) in enumerate(sample_div):
    print(f"Fold {i}: Train shape = {train_df.shape}, Test shape = {test_df.shape}")

        




    










    


