import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import helper_funcs as hf

def k_fold_partition(tsv_file_path, k):
    df = hf.load_tsv_file(tsv_file_path)
    # Use np.array_split to get indices, then use iloc to get DataFrames
    indices = np.array_split(np.arange(len(df)), k)
    list_of_dfs = [df.iloc[idx].reset_index(drop=True) for idx in indices]
    return list_of_dfs

def div_test_train_sets(list_of_dfs):

    num_sets = len(list_of_dfs)

    

sample = k_fold_partition(tsv_file_path='projectData/chr1_200bp_bins.tsv', k=10)

# Now 'sample' is a list of k DataFrames
for i, df_part in enumerate(sample):
    print(df_part.shape)


    










    


