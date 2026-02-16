import pandas as pd
import numpy as np

def confusion_matrix(proc_df,tf_id):
    """Builds confusion matrix from processed dataframe
    Args:
        proc_df (DataFrame) : Processed Dataframe with actual and predicted classes
        tf_id (str) : Transcription Factor ID column name
    Output:
        conf_matrix (dict) : Confusion Matrix with TP, FP, TN, FN counts
    """

    actual = proc_df[tf_id].values
    pred = proc_df[f'Pred_{tf_id}'].values

    TP = ((actual == 'B') & (pred == 'B')).sum()
    FP = ((actual == 'U') & (pred == 'B')).sum()
    TN = ((actual == 'U') & (pred == 'U')).sum()
    FN = ((actual == 'B') & (pred == 'U')).sum()

    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

def precision(conf_matrix):
    return conf_matrix['TP']/(conf_matrix['TP']+conf_matrix['FP'])

def recall(conf_matrix):
    return conf_matrix['TP']/(conf_matrix['TP']+conf_matrix['FN'])

def specificity(conf_matrix):
    return conf_matrix['TN']/(conf_matrix['TN']+conf_matrix['FP'])


def thresholds(score_df,markov_order):
    """
    Generates sorted unique thresholds from dataframe with scores.
    Args:
        score_df (DataFrame) : DataFrame containing scores
        markov_order (int) : Markov Order for score column
    Output:
        thresholds (ndarray) : Sorted unique thresholds
    """
    return np.sort(score_df[f'Score_{markov_order}'].unique())

def classification_results(sorted_df,threshold,tf_id,markov_order):
    """
    Classifies regions as U or B for a given threshold.
    Args:
        sorted_df (DataFrame) : DataFrame containing scores and actual classes
        threshold (float) : Threshold for classification
        tf_id (str) : Transcription Factor ID column name
        markov_order (int) : Markov Order for score column
    Output:
        results (dict) : Dictionary containing Precision, Recall and Specificity
    """

    scores = sorted_df[f'Score_{markov_order}'].values
    actuals = sorted_df[tf_id].values

    preds = np.where(scores >= threshold, 'B', 'U')

    #print("Building confusion matrix...")
    TP = ((actuals == 'B') & (preds == 'B')).sum()
    FP = ((actuals == 'U') & (preds == 'B')).sum()
    TN = ((actuals == 'U') & (preds == 'U')).sum()
    FN = ((actuals == 'B') & (preds == 'U')).sum()

    conf_matrix = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

    results = { 'Precision' : precision(conf_matrix=conf_matrix),
               'Recall' : recall(conf_matrix=conf_matrix),
                'Specificity' : specificity(conf_matrix=conf_matrix) }
    
    return results


def prec_rec_spec(test_res_df,chr_id,tf_id,markov_order):

    """
    Generates Precision-Recall-Specificity dataframe from test results.
    Args:
        test_res_df (DataFrame) : DataFrame containing test results with scores
        chr_id (str) : Chromosome ID
        tf_id (str) : Transcription Factor ID column name
        markov_order (int) : Markov Order for score column
    Output:
        prs_df (DataFrame) : DataFrame containing Precision, Recall and Specificity at various thresholds
    """

    p_thresholds = thresholds(score_df=test_res_df,
                              markov_order=markov_order)
    
    if len(p_thresholds)>2000:
        p_thresholds = np.linspace(p_thresholds.min(), p_thresholds.max(),2000)
    
    prs_df = pd.DataFrame()

    prec_list = []
    recl_list = []
    spec_list = []

    for trld in p_thresholds:

        results = classification_results(test_res_df,
                                         threshold=trld,
                                         tf_id=tf_id,
                                         markov_order=markov_order)
        
        prec_list.append(results['Precision'])
        recl_list.append(results['Recall'])
        spec_list.append(results['Specificity'])

    prs_df['Threshold'] = p_thresholds
    prs_df['Precision'] = prec_list
    prs_df['Recall'] = recl_list
    prs_df['Specificity'] = spec_list

    return prs_df


def AU_PRC(prs_vals,save=None):
    """
    Calculates Area Under Precision-Recall Curve
    Args:
        prs_vals (DataFrame) : DataFrame containing Precision and Recall values
    Output:
        au_prc (float) : Area Under Precision-Recall Curve
    """

    prec_vals = prs_vals['Precision'].to_numpy()
    recall_vals = prs_vals['Recall'].to_numpy()

    sorted_indices = np.argsort(recall_vals)
    return np.trapz(prec_vals[sorted_indices], recall_vals[sorted_indices])

def AU_ROC(prs_vals,save=None):
    """
    Calculates Area Under Receiver-Operating Characteristic Curve
    Args:
        prs_vals (DataFrame) : DataFrame containing Specificity and Recall values
    Output:
        au_roc (float) : Area Under Receiver Operating Characteristic Curve
    """
    recall_vals = prs_vals['Recall'].to_numpy()
    spec_vals = prs_vals['Specificity'].to_numpy()
    fpr = 1 - spec_vals

    sorted_indices = np.argsort(fpr)
    return np.trapz(recall_vals[sorted_indices], fpr[sorted_indices])