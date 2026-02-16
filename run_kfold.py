import argparse
import kfold_cv
import fig_funcs
import os
import time

"""
Execution File for the whole program.
Run from the terminal 
"""
TSV_PATH = 'projectData/chr1_200bp_bins.tsv'
FASTA_PATH = 'projectData/chr1.fa'

try:
    os.mkdir("./resultData")
    print("created result data directory")
except FileExistsError:
    print("Directory 'resultsData' already exists.")
except FileNotFoundError:
    print("Parent directory does not exist.")

def main():
    parser = argparse.ArgumentParser(description="Run k-fold CV with Markov model")

    parser.add_argument("--tf_id", required=True, help="Transcription factor ID (REST,EP300,CTCF))")
    parser.add_argument("--chr_id", required=True, help="Chromosome ID chr<n>")
    parser.add_argument("--markov_order", type=int, required=True, help="Markov order (0-10)")
    parser.add_argument("--kfold", type=int, required=True, help="k-fold value (3-5)")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs")

    args = parser.parse_args()
    start_time = time.time()
    try:
        final_results = kfold_cv.run_kfold_parallel(
            tsv_path=f'projectData/{args.chr_id}_200bp_bins.tsv',
            fasta_path=f'projectData/{args.chr_id}.fa',
            tf_id=args.tf_id,
            chr_id=args.chr_id,
            markov_order=args.markov_order,
            k=args.kfold,
            num_cpus=args.num_cpus
        )
    except Exception as e:
        print(f"Error for markov_order={args.markov_order}, kfold={args.kfold}: {e}")
        return
    end_time = time.time()
    print(f"Time Taken to calculate AUC and PRC : {end_time-start_time}s", )
    fig_funcs.plot_auPRC(
        markov_order=args.markov_order,
        kfold=args.kfold,
        plots_file_path=f"./{args.tf_id}PRC_PLOTS"
    )

    fig_funcs.plot_auROC(
        markov_order=args.markov_order,
        kfold=args.kfold,
        plots_file_path=f"./{args.tf_id}_ROC_PLOTS"
    )


if __name__ == "__main__":
    main()
