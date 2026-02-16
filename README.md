Codebase for the Computational Functional Genomics course offered in January 2026 at IISER Pune.

The goal of the project is to build a classifier to identify regions in the genome that are binding sites for transcription factors(TFs).

The first part of the project involves building a Markov-based classifier and testing it via k-fold cross validation. We use Precision-Recall and Receiver-Operator metrics.

#### Dependencies
Python (3.13.7)
- NumPy (2.2.6)
- PanDas (3.0.0)
- MatPlotlib (3.10.3)
- tqdm (4.67.3)
- pyfaidx (0.9.0.3)

#### Data

Since whole genome data is too huge to upload here, you'll have to download the FASTA files.
Here's how to do that:

For MacOS
- `curl -o projectData/chr1.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz`
- `gunzip projectData/chr1.fa.gz` 
(replace chr1 with whichever chromosome from 1-19)

For Linux
- `wget -P projectData/ https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz`
- `gunzip projectData/chr1.fa.gz`

For Windows
- Reconsider your choices.

#### Running the code

The code can be executed via the terminal using the command:
```python run_kfold.py --tf_id [TF_ID] --chr_id [CHR NUMBER] --markov_order [M] --kfold [K] --num_cpus [#CORES USED] ```
- `chr_id` has to be of the form `chr<n>` where n is a number in [1,19].
- `tf_id` has to one of [REST, EP300, CTCF].
- `kfold` has to be a number in [1,10].
- `markov_order` has to a number in [0,10].

.csv files containing data on sensitivity, specificity, precision will be stored in `/resultsData`
`/projectData` contains FASTA files for the chromosomes as well as bound-unbound information for each chromosome segrated into bins.

Precision-Recall Curves will be stored in `./PRC_PLOTS` and Receiever-Operator Curves in `./ROC_PLOTS`

#### Data

FASTA files for each chromosome can be downloaded from [the UCSC genome browser](https://hgdownload.gi.ucsc.edu/goldenPath/hg38/chromosomes/).






