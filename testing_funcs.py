import pandas as pd
import numpy as np

def confusion_matrix(proc_df,tf_id):

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
    return np.sort(score_df[f'Score_{markov_order}'].unique())

def classification_results(sorted_df,threshold,tf_id,markov_order):

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

def precision_recall(test_res_df,chr_id,tf_id,markov_order):

    p_thresholds = thresholds(score_df=test_res_df,
                              markov_order=markov_order)
    
    if len(p_thresholds) > 2000:
        p_thresholds = np.linspace(p_thresholds.min(), p_thresholds.max(), 2000)
    
    pr_df = pd.DataFrame()

    prec_list = []
    recl_list = []

    for trld in p_thresholds:
        results = classification_results(test_res_df,
                                         threshold=trld,
                                         tf_id=tf_id,
                                         markov_order=markov_order)
        prec_list.append(results['Precision'])
        recl_list.append(results['Recall'])

    pr_df['Threshold'] = p_thresholds
    pr_df['Precision'] = prec_list
    pr_df['Recall'] = recl_list

    return pr_df

def prec_rec_spec(test_res_df,chr_id,tf_id,markov_order):

    #print("Calculating Precision, Recall and Specificity...")

    p_thresholds = thresholds(score_df=test_res_df,
                              markov_order=markov_order)
    
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

def reciever_operator(test_res_df,chr_id,tf_id,markov_order):

    p_thresholds = thresholds(score_df=test_res_df,
                              markov_order=markov_order)
    
    ro_df = pd.DataFrame()

    sens_list = []
    spec_list = []

    for trld in p_thresholds:

        results = classification_results(test_res_df,
                                         threshold=trld,
                                         tf_id=tf_id,
                                         markov_order=markov_order)
        
        sens_list.append(results['Recall'])
        spec_list.append(1-results['Specificity'])

    ro_df['Threshold'] = p_thresholds
    ro_df['Precision'] = sens_list
    ro_df['Recall'] = spec_list

    return ro_df


def AU_PRC(prs_vals):
    #print("Calculating auPRC...")

    prec_vals = prs_vals['Precision'].to_numpy()
    recall_vals = prs_vals['Recall'].to_numpy()

    sorted_indices = np.argsort(recall_vals)
    return np.trapezoid(prec_vals[sorted_indices], recall_vals[sorted_indices])

def AU_ROC(prs_vals):
    #print("Calculating auROC...")
    recall_vals = prs_vals['Recall'].to_numpy()
    spec_vals = prs_vals['Specificity'].to_numpy()
    fpr = 1 - spec_vals

    sorted_indices = np.argsort(fpr)
    return np.trapezoid(recall_vals[sorted_indices], fpr[sorted_indices])