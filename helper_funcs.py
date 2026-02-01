import numpy as np
from tqdm import tqdm
from time import time
import pandas as pd
from pyfaidx import Fasta
import itertools

def load_tsv_file(file_path):
    """
    Converts .tsv file into a PanDas dataframe
    Args:
        file_path : File Path of .tsv file
    Output:
        chrom_df : Dataframe
    """
    chrom_df = pd.read_csv(f"{file_path}",sep='\t')
    return chrom_df

def generate_in_seqs(markov_order):
    bases = ['A','T','G','C']
    seqs = []

    seq_list = itertools.product(bases, repeat=markov_order)
    for seq in seq_list:
        seqs.append(''.join(seq))

    return seqs

def count_in_region(markov_order,file_path,chr_id,str_idx,end_idx):

    inp_bases = generate_in_seqs(markov_order)
    out_bases = ['A','T','G','C']

    counts = {f'{i_base}_{o_base}' : 0 for i_base in inp_bases for o_base in out_bases}
    
    region = Fasta(f'{file_path}')

    for i in range(str_idx-1,end_idx,markov_order):

        in_bite = str(region[f'{chr_id}'][i:i+markov_order]).upper()
        out_bite = str(region[f'{chr_id}'][i+markov_order:i+markov_order+1]).upper()

        if ('N' in in_bite) or ('N' in out_bite):
            continue

        key = f'{in_bite}_{out_bite}'
        if key in counts:
            counts[key] += 1
            
    return counts

"""
def stripped_df(tsv_file_path, 
                tf_id,
                bclass, #None if you want both U and B, specify if you want only one
                tf_list = ['EP300','CTCF','ATAC','REST']):
    
    df = load_tsv_file(tsv_file_path)
    if bclass != None:
        target_df = df[df[f'{tf_id}'] == f'{bclass}']
    else:
        target_df = df
    
    cols_to_drop = [tf for tf in tf_list if tf != tf_id]

    target_df = target_df.drop(columns=cols_to_drop)

    return target_df
"""

def stripped_df(df, 
                tf_id,
                bclass, #None if you want both U and B, specify if you want only one
                tf_list = ['EP300','CTCF','ATAC','REST']):
    
    if bclass != None:
        target_df = df[df[f'{tf_id}'] == f'{bclass}']
    else:
        target_df = df
    
    cols_to_drop = [tf for tf in tf_list if tf != tf_id]

    target_df = target_df.drop(columns=cols_to_drop)

    return target_df

def construct_transition_matrix(markov_order,
                                fasta_file_path, 
                                target_df, 
                                chr_id,
                                tf_id, #Name of Transcription Factor
                        
                                tf_list=['EP300','CTCF','ATAC','REST']):


    inp_bases = generate_in_seqs(markov_order)
    out_bases = ['A','T','G','C']

    total_counts = {f'{i_base}_{o_base}' : 0 for i_base in inp_bases for o_base in out_bases}

    for row in tqdm(target_df.itertuples(), total=len(target_df), desc="Constructing Transition Matrices"):
        start = row.start
        end = row.end

        temp_counts = count_in_region(markov_order=markov_order,
                                      file_path=fasta_file_path,
                                      chr_id=chr_id,
                                      str_idx=start,
                                      end_idx=end)
        
        for key in temp_counts.keys():
            total_counts[key] += temp_counts[key]

    for ibase in inp_bases:
        total = 0
        for obase in out_bases:
            total += total_counts[f'{ibase}_{obase}']

        for obase in out_bases:
            total_counts[f'{ibase}_{obase}'] = (total_counts[f'{ibase}_{obase}']+1)/(total + 4) #Pseudocounts added

    return total_counts


def log_odds_single(inbase,outbase, bmatrix, umatrix):

    """
    Returns log odd score for a single transition whatever
    """

    b_prob = bmatrix[f'{inbase}_{outbase}']
    u_prob = umatrix[f'{inbase}_{outbase}']

    return np.log(b_prob/u_prob)

def log_odds_total(markov_order, fasta_file_path, chr_id, str_idx, end_idx, bmatrix, umatrix):

    score = 0.0

    region = Fasta(f"{fasta_file_path}")

    for i in range(str_idx-1,end_idx,markov_order):

        in_bite = str(region[f'{chr_id}'][i:i+markov_order]).upper()
        out_bite = str(region[f'{chr_id}'][i+markov_order:i+markov_order+1]).upper()

        if ('N' in in_bite) or ('N' in out_bite):
            continue

        score += log_odds_single(inbase=in_bite,outbase=out_bite,bmatrix=bmatrix,umatrix=umatrix)

    return score

def binding_prob_database(markov_order,tf_data,fasta_file_path,chr_id,bmatrix,umatrix):

    bprob_df = tf_data.copy(deep=True)

    bprobs_list = []

    for row in tqdm(tf_data.itertuples(), total=len(tf_data), desc="Calculating Scores"):

        start = row.start
        end = row.end

        bprobs_list.append(log_odds_total(markov_order=markov_order,
                                          fasta_file_path=fasta_file_path,
                                          chr_id=chr_id,
                                          str_idx=start,
                                          end_idx=end,
                                          bmatrix=bmatrix,
                                          umatrix=umatrix))
        
    bprob_df[f'Score_{markov_order}'] = bprobs_list

    return bprob_df










    

    


 

















    




















































    






