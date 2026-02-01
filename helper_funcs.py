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
    """
    Generates all possible combinations of 'from' sequences of given Markov order
    Args:
        markov_order(int) : Order of Markov Model 
    Output:
        seqs (list) : List of all possible in-sequences of given order
    """

    bases = ['A','T','G','C']
    seqs = []

    seq_list = itertools.product(bases, repeat=markov_order)
    for seq in seq_list:
        seqs.append(''.join(seq))

    return seqs

def count_in_region(markov_order,fasta_obj,chr_id,str_idx,end_idx):

    """
    Counts occurrences of each possible transition in a given region
    Args:
        markov_order(int) : Order of Markov Model
        fasta_obj (Fasta Object) : Fasta object of genome
        chr_id (str) : Chromosome ID
        str_idx (int) : Start Index of region (1-based)
        end_idx (int) : End Index of region (1-based, inclusive)
    Output:
        counts (dict) : Dictionary of counts of each transition in the region
    """

    inp_bases = generate_in_seqs(markov_order)
    out_bases = ['A','T','G','C']

    counts = {f'{i_base}_{o_base}' : 0 for i_base in inp_bases for o_base in out_bases}
    

    try:
        full_seq = str(fasta_obj[f'{chr_id}'][str_idx-1:end_idx]).upper()
    except KeyError:
        return counts

    seq_len = len(full_seq)

    for i in range(seq_len - markov_order):

        in_bite = full_seq[i:i+markov_order]
        out_bite = full_seq[i+markov_order]

        if ('N' in in_bite) or ('N' in out_bite):
            continue

        key = f'{in_bite}_{out_bite}'
        if key in counts:
            counts[key] += 1
            
    return counts

def stripped_df(df, 
                tf_id,
                bclass, #None if you want both U and B, specify if you want only one
                tf_list = ['EP300','CTCF','ATAC','REST']):
    
    """
    Strips dataframe to only necessary columns as specified by tf_id and bclass
    Args:
        df (DataFrame) : Original DataFrame
        tf_id (str) : Transcription Factor ID -> enter the TF you want to analyze
        bclass (str/None) : 'B' or 'U' or None -> specify if you want only regions where the TF is bound/unbound
        tf_list (list) : List of all Transcription Factors in dataset
    Output:
        target_df (DataFrame) : Stripped DataFrame
    """
    
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
    """
    Constructs Transition Matrix for given Markov order from target dataframe
    Args:
        markov_order (int) : Order of Markov Model
        fasta_file_path (str) : File Path of Fasta file
        target_df (DataFrame) : DataFrame containing regions to build matrix from
        chr_id (str) : Chromosome ID
        tf_id (str) : Transcription Factor ID
        tf_list (list) : List of all Transcription Factors in dataset
    Output:
        total_counts (dict) : Dictionary of transition probabilities
    """


    inp_bases = generate_in_seqs(markov_order)
    out_bases = ['A','T','G','C']

    total_counts = {f'{i_base}_{o_base}' : 0 for i_base in inp_bases for o_base in out_bases}

    fasta_obj = Fasta(f'{fasta_file_path}')

    for row in tqdm(target_df.itertuples(), total=len(target_df), desc="Constructing Transition Matrices"):
        start = row.start
        end = row.end

        temp_counts = count_in_region(markov_order=markov_order,
                                      fasta_obj=fasta_obj,
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

def build_score_dict(bmatrix, umatrix):
    """
    Dictionary for faster lookup for log-odds scores
    Args:
        bmatrix (dict) : Transition probability matrix for Bound regions
        umatrix (dict) : Transition probability matrix for Unbound regions
    Output:
        score_dict (dict) : Dictionary of log-odds scores for each transition
    """
    score_dict = {}
    # Iterate over all keys present in the matrices
    for key in bmatrix.keys():
        b_prob = bmatrix.get(key, 1e-10) # Small float safety
        u_prob = umatrix.get(key, 1e-10)
        # Pre-calculate the log difference
        score_dict[key] = np.log(b_prob / u_prob)
    return score_dict

def log_odds_single(inbase,outbase, bmatrix, umatrix):

    """
    Returns log odd score for a single transition whatever
    Args:
        inbase (str) : 'From' sequence
        outbase (str) : 'To' base
        bmatrix (dict) : Transition probability matrix for Bound regions
        umatrix (dict) : Transition probability matrix for Unbound regions
    Output:
        log_odds (float) : Log-Odds Score
    """

    b_prob = bmatrix.get(f'{inbase}_{outbase}', 1e-6) # Added safety get
    u_prob = umatrix.get(f'{inbase}_{outbase}', 1e-6)
    return np.log(b_prob/u_prob)

def log_odds_total(markov_order, fasta_obj, chr_id, str_idx, end_idx, score_dict):
    """
    Calculates total log-odds score for a given region
    Args:
        markov_order (int) : Order of Markov Model
        fasta_obj (Fasta Object) : Fasta object of genome
        chr_id (str) : Chromosome ID
        str_idx (int) : Start Index of region (1-based)
        end_idx (int) : End Index of region (1-based, inclusive)
        score_dict (dict) : Dictionary of log-odds scores for each transition
    Output:
        score (float) : Total log-odds score for the region
    """

    score = 0.0

    try:
        full_seq = str(fasta_obj[f'{chr_id}'][str_idx-1:end_idx]).upper()
    except KeyError:
        return 0.0
        
    
    seq_len = len(full_seq)

    for i in range(seq_len - markov_order):

        in_bite = full_seq[i : i + markov_order]
        out_bite = full_seq[i + markov_order]

        if ('N' in in_bite) or ('N' in out_bite):
            continue

        score += score_dict.get(f'{in_bite}_{out_bite}', 0.0)

    return score

def binding_prob_database(markov_order,tf_data,fasta_file_path,chr_id,bmatrix,umatrix):
    """ 
    Builds a database of binding probabilities for all regions in tf_data
    Args:
        markov_order (int) : Order of Markov Model
        tf_data (DataFrame) : DataFrame containing regions to score
        fasta_file_path (str) : File Path of Fasta file
        chr_id (str) : Chromosome ID
        bmatrix (dict) : Transition probability matrix for Bound regions
        umatrix (dict) : Transition probability matrix for Unbound regions
    Output:
        bprob_df (DataFrame) : DataFrame with added column of binding scores
    """

    bprob_df = tf_data.copy(deep=True)
    bprobs_list = []

    fasta_obj = Fasta(f"{fasta_file_path}")

    score_dict = build_score_dict(bmatrix, umatrix)

    for row in tqdm(tf_data.itertuples(), total=len(tf_data), desc="Calculating Scores"):

        start = row.start
        end = row.end

        bprobs_list.append(log_odds_total(markov_order=markov_order,
                                          fasta_obj=fasta_obj,
                                          chr_id=chr_id,
                                          str_idx=start,
                                          end_idx=end,
                                          score_dict=score_dict))
        
    bprob_df[f'Score_{markov_order}'] = bprobs_list

    return bprob_df










    

    


 

















    




















































    






