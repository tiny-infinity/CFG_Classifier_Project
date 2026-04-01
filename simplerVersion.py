import argparse
import math
from collections import defaultdict
from tqdm import tqdm

def read_fasta(file_path):
    """Reads a FASTA file and returns a list of sequence strings."""
    print("Reading FASTA files...")
    sequences = []
    with open(file_path, 'r') as f:
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    sequences.append("".join(seq).upper())
                    seq = []
            else:
                seq.append(line)
        if seq:
            sequences.append("".join(seq).upper())
    return sequences

def train_markov_model(sequences, m):
    """
    Trains an order m Markov model on a list of sequences.
    Returns a dictionary of log2 probabilities.
    """
    print("Training Markov models...")
    counts = defaultdict(lambda: {'A': 1, 'C': 1, 'G': 1, 'T': 1})
    
    for seq in tqdm(sequences, desc=f"Training Order {m} Model"):
        for i in range(len(seq) - m):
            context = seq[i:i+m]
            next_base = seq[i+m]
            
            if next_base in "ACGT" and all(b in "ACGT" for b in context):
                counts[context][next_base] += 1
    log_probs = {}
    for context, next_counts in counts.items():
        total_transitions = sum(next_counts.values())
        log_probs[context] = {
            base: math.log2(count / total_transitions) 
            for base, count in next_counts.items()
        }
        
    return log_probs

def score_sequences(sequences, log_probs, m):
    """
    Calculates the log-likelihood score for each sequence based on the trained model.
    """
    default_log_prob = math.log2(0.25)
    print("Scoring Sequences...")
    
    for seq in tqdm(sequences,desc="SCORING SEQUENCES..."):
        score = 0.0
        
        
        for i in range(len(seq) - m):
            context = seq[i:i+m]
            next_base = seq[i+m]
            
            if next_base not in "ACGT" or not all(b in "ACGT" for b in context):
                continue 
                
            if context in log_probs and next_base in log_probs[context]:
                score += log_probs[context][next_base]
            else:
                score += default_log_prob
                
        
        print(score)

def main():
    parser = argparse.ArgumentParser(description="Train and score a FASTA file using a Markov Model.")
    parser.add_argument("--fasta_file", required=True, help="Path to the input FASTA file")
    parser.add_argument("--m", type=int,required=True, help="Order of the Markov Model (m)")
    
    args = parser.parse_args()
    
    try:
        sequences = read_fasta(args.fasta_file)
        
        if not sequences:
            print("Error: No sequences found in the FASTA file.")
            return

        log_probs = train_markov_model(sequences, args.m)
        
        score_sequences(sequences, log_probs, args.m)
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{args.fasta_file}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()