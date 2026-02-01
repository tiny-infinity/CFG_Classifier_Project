import kfold_cv

# CONFIGURATION
TSV_PATH = 'projectData/chr1_200bp_bins.tsv'
FASTA_PATH = 'projectData/chr1.fa'
TF_ID = 'REST'
CHR_ID = 'chr1'

markov_orders = [i for i in range(0,11,1)]
k_vals = [3,4,5]

for mo in markov_orders:
    for k in k_vals:
        try:
            final_results = kfold_cv.run_kfold_parallel(tsv_path=TSV_PATH, 
                                               fasta_path=FASTA_PATH,
                                               tf_id=TF_ID, 
                                               chr_id=CHR_ID, 
                                               markov_order=mo, 
                                               k=k)
        except Exception as e:
            print(f"An error occurred for markov_order={mo}, k={k}: {e}")



