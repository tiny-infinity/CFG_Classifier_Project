import pandas as pd
import numpy as np

def confusion_matrix(proc_df,tf_id):

    conf_matrix = {'TP':0,'FP':0,'TN':0,'FN':0}

    for row in proc_df.itertuples():
        actual = getattr(row, tf_id)
        pred = getattr(row, f'Pred_{tf_id}')
        if actual == 'B' and pred == 'B':
            conf_matrix['TP'] += 1
        elif actual == 'U' and pred == 'B':
            conf_matrix['FP'] += 1
        elif actual == 'U' and pred == 'U':
            conf_matrix['TN'] += 1
        elif actual == 'B' and pred == 'U':
            conf_matrix['FN'] += 1

    return conf_matrix

def precision(conf_matrix):
    return conf_matrix['TP']/(conf_matrix['TP']+conf_matrix['FP'])

def recall(conf_matrix):
    return conf_matrix['TP']/(conf_matrix['TP']+conf_matrix['FN'])

def specificity(conf_matrix):
    return conf_matrix['TN']/(conf_matrix['TN']+conf_matrix['FP'])


def thresholds(score_df,markov_order,tf_id):
    sorted_df = score_df.sort_values(by=f'Score_{markov_order}')
    return sorted_df[f'Score_{markov_order}'].to_numpy()

def classification_results(sorted_df,threshold,tf_id,markov_order):

    res_df = sorted_df.copy(deep=True)
    res_df[f'Pred_{tf_id}'] = np.where(res_df[f'Score_{markov_order}']>=threshold, 'B', 'U')

    conf_matrix = confusion_matrix(proc_df=res_df,tf_id=tf_id)

    results = { 'Precision' : precision(conf_matrix=conf_matrix),
               'Recall' : recall(conf_matrix=conf_matrix),
                'Specificity' : specificity(conf_matrix=conf_matrix) }
    
    return results

def precision_recall(test_res_df,chr_id,tf_id,markov_order):

    p_thresholds = thresholds(score_df=test_res_df,
                              chr_id = chr_id,
                              tf_id = tf_id,
                              markov_order=markov_order)
    
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

    p_thresholds = thresholds(score_df=test_res_df,
                              chr_id = chr_id,
                              tf_id = tf_id,
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
    prs_df['Specificity'] = specificity

    return prs_df

def reciever_operator(test_res_df,chr_id,tf_id,markov_order):

    p_thresholds = thresholds(score_df=test_res_df,
                              chr_id = chr_id,
                              tf_id = tf_id,
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

    prec_vals = prs_vals['Precision'].to_numpy()
    recall_vals = prs_vals['Recall'].to_numpy()

    auprc = np.trapezoid(prec_vals,recall_vals)

    return auprc

def AU_ROC(prs_vals):

    recall_vals = prs_vals['Recall'].to_numpy()
    spec_vals = prs_vals['Specificity'].to_numpy()

    auroc = np.trapezoid(recall_vals,spec_vals)

    return auroc