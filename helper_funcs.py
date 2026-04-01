import numpy as np
from tqdm import tqdm
from time import time
import pandas as pd
from pyfaidx import Fasta
import itertools
import pyBigWig
from collections import defaultdict

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

def bin_start_end_coords(base_df):

    """
    Extracts start coordinates and end coordinates from the bins dataframe
    """

    starts = base_df['start'].tolist()
    ends = base_df['end'].tolist()

    return starts, ends

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

def map_fimo_to_bins(fimo_tsv_path,base_df,tf_id):

    """
    Assigns MEME-FIMO score to each 200bp bin
    Args:
        base_df : Dataframe containing bins
        fimo_tsv_path : File path for the MEME-FIMO .tsv file
        tf_id : Name of TF

    Output:
        base_df : Dataframe with MEME-FIMO scores added for each bin.
    """

    fimo_hits = pd.read_csv(fimo_tsv_path,sep='\t',comment="#")
    fimo_hits['temp_idx'] = fimo_hits['start']//200
    best_hits_map = fimo_hits.groupby('temp_idx')['score'].max().to_dict()   
    temp_base_idx = base_df['start'] // 200
    base_df[f'FIMO_{tf_id}'] = temp_base_idx.map(best_hits_map).fillna(0.0)

    return base_df

def phastcons_to_bins(pc_file_path,starts,ends,chr_id):

    """
    Assigns PhastCons conservation scores to each 200bp bin
    Args:
        pc_file_path : File path for PhastCons BigWig file
        starts : coordinates of the starting points of each bin
        ends : coordinates of the ending points of each bin
        chr_id : name of chromosome (e.g chr1)

    Output:
        base_df : Dataframe with PhastCons scores added
    """

    bw = pyBigWig.open(pc_file_path)
    bin_means = []

    for s,e in zip(starts,ends):
        val = bw.stats(chr_id,int(s),int(e),type='mean')[0]
        bin_means.append(val if val is not None else 0.0)

    bw.close()

    return bin_means

def count_in_region(markov_order, fasta_obj, chr_id, str_idx, end_idx):
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

    counts = defaultdict(int)

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

        counts[f'{in_bite}_{out_bite}'] += 1

    return counts


def build_global_transition_matrix(chr_set_ids,tf_id,markov_order):
    global_B_counts = defaultdict(int)
    global_U_counts = defaultdict(int)

    for chr_num in chr_set_ids:

        fasta_obj = Fasta(f"projectData/chr{chr_num}.fa")
        print(f"Reading through chromosome number {chr_num}")

        bin_tsv_path = f"projectData/chr{chr_num}_200bp_bins.tsv"

        base_df = pd.read_csv(bin_tsv_path,sep='\t')
    
        bound_df = base_df[base_df[f'{tf_id}']=='B']
        unbound_df = base_df[base_df[f'{tf_id}']=='U']

        for row in bound_df.itertuples():
            temp_counts = count_in_region(markov_order,chr_id=f'chr{chr_num}',fasta_obj=fasta_obj,str_idx=row.start,end_idx=row.end)

            for key,val in temp_counts.items():
                global_B_counts[key]+=val

        for row in unbound_df.itertuples():
            temp_counts = count_in_region(markov_order,chr_id=f'chr{chr_num}',fasta_obj=fasta_obj,str_idx=row.start,end_idx=row.end)

            for key,val in temp_counts.items():
                global_U_counts[key]+=val

    inp_bases = generate_in_seqs(markov_order)
    out_bases = ['A', 'T', 'G', 'C']
    
    prob_B = {}
    prob_U = {}

    for ibase in inp_bases:
        # Calculate totals for THIS specific prefix across all chromosomes
        Btotal = sum(global_B_counts[f'{ibase}_{obase}'] for obase in out_bases)
        Utotal = sum(global_U_counts[f'{ibase}_{obase}'] for obase in out_bases)

        for obase in out_bases:
            key = f'{ibase}_{obase}'
            # Apply pseudocounts (+1 / +4) to the global sums
            prob_B[key] = (global_B_counts[key] + 1) / (Btotal + 4)
            prob_U[key] = (global_U_counts[key] + 1) / (Utotal + 4)

    return prob_B, prob_U

def construct_transition_matrix(markov_order,
                                fasta_file_path, 
                                target_df, 
                                chr_id,
                                tf_id,
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

    total_counts = defaultdict(int)
    fasta_obj = Fasta(fasta_file_path)

    for row in tqdm(target_df.itertuples(), total=len(target_df)):
        temp_counts = count_in_region(markov_order, fasta_obj, chr_id, row.start, row.end)

        for key, val in temp_counts.items():
            total_counts[key] += val

    # convert to probabilities with pseudocounts
    for ibase in inp_bases:
        total = sum(total_counts[f'{ibase}_{obase}'] for obase in out_bases)

        for obase in out_bases:
            key = f'{ibase}_{obase}'
            total_counts[key] = (total_counts[key] + 1) / (total + 4)

    return dict(total_counts)


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
    
    for key in bmatrix.keys():
        b_prob = bmatrix.get(key, 1e-10) 
        u_prob = umatrix.get(key, 1e-10)
        
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

def binding_prob_database(markov_order,tf_data,tf_id,fasta_file_path,chr_id,bmatrix,umatrix):
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
        
    bprob_df[f'log_odds{tf_id}'] = bprobs_list

    return bprob_df


def assign_global_log_odds(chrom_set,tf_id,markov_order):
    #creating the transition matrix using all chromosomes in chrom_set
    Bmatrix, Umatrix = build_global_transition_matrix(chr_set_ids=chrom_set,markov_order=markov_order,tf_id=tf_id)

    processed_dfs = {}

    for chr_num in chrom_set:
        
        tsv_file_path = f"projectData/chr{chr_num}_200bp_bins.tsv"
        base_df = pd.read_csv(tsv_file_path,sep='\t')

        fasta_file_path = f"projectData/chr{chr_num}.fa"

        processed_dfs[f'chr{chr_num}']=binding_prob_database(markov_order=markov_order,tf_data=base_df,fasta_file_path=fasta_file_path,
                                                   chr_id=f"chr{chr_num}",bmatrix=Bmatrix,umatrix=Umatrix,tf_id=tf_id)
        
    return processed_dfs
        

def build_feature_matrix(chrom_set,tf_id,markov_order):

    processed_dfs = assign_global_log_odds(chrom_set, tf_id,markov_order) #returns dictionary of {'chr_id' : 'dataframe with log odds scores'}
    final_list = []

    for chr_num in chrom_set:
        print(f"Building Features for Chromosome {chr_num}")
        df_with_fimo = map_fimo_to_bins(fimo_tsv_path=f'projectData/{tf_id}_FIMO_RESULTS/chr{chr_num}_fimo/fimo.tsv',
                                        base_df = processed_dfs[f'chr{chr_num}'],
                                        tf_id=tf_id)
        processed_dfs[f'chr{chr_num}'] = df_with_fimo #adds column of FIMO scores

        starts = processed_dfs[f'chr{chr_num}']['start'].tolist()
        ends = processed_dfs[f'chr{chr_num}']['end'].tolist()

        processed_dfs[f'chr{chr_num}']['PhastCons'] = phastcons_to_bins(pc_file_path='projectData/hg38.phastCons7way.bw',
                                                                       starts=starts,
                                                                       ends=ends,
                                                                       chr_id=f'chr{chr_num}')#adds column of PhastCons scores

        processed_dfs[f'chr{chr_num}']['PhastCons'] =  processed_dfs[f'chr{chr_num}']['PhastCons'].astype('float32')
        processed_dfs[f'chr{chr_num}'][f'FIMO_{tf_id}'] =  processed_dfs[f'chr{chr_num}'][f'FIMO_{tf_id}'].astype('float32')
        processed_dfs[f'chr{chr_num}'][f'log_odds{tf_id}'] =  processed_dfs[f'chr{chr_num}'][f'log_odds{tf_id}'].astype('float32')

        keep_cols = ['start','end','ATAC',f'log_odds{tf_id}',f'FIMO_{tf_id}','PhastCons',f'{tf_id}']

        if tf_id == 'EP300':
            processed_dfs[f'chr{chr_num}'] = map_fimo_to_bins(f'projectData/CTCF_FIMO_RESULTS/chr{chr_num}_fimo/fimo.tsv', processed_dfs[f'chr{chr_num}'], 'CTCF')
            processed_dfs[f'chr{chr_num}'] = map_fimo_to_bins(f'projectData/REST_FIMO_RESULTS/chr{chr_num}_fimo/fimo.tsv', processed_dfs[f'chr{chr_num}'], 'REST')
            processed_dfs[f'chr{chr_num}'] = map_fimo_to_bins(f'projectData/FOXA1_FIMO_RESULTS/chr{chr_num}_fimo/fimo.tsv', processed_dfs[f'chr{chr_num}'], 'CTCF')
            processed_dfs[f'chr{chr_num}'] = map_fimo_to_bins(f'projectData/GATA3_FIMO_RESULTS/chr{chr_num}_fimo/fimo.tsv', processed_dfs[f'chr{chr_num}'], 'REST')

        final_list.append(processed_dfs[f'chr{chr_num}'][keep_cols])

    master_df = pd.concat(final_list,axis=0).reset_index(drop=True)
    return master_df

















        









        












    

    


 

















    




















































    






