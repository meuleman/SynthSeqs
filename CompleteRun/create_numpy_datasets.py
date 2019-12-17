import numpy as np
import pandas as pd
from Bio import SeqIO

PATH_TO_REFERENCE_GENOME = \
    "/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa"

PATH_TO_DHS_MASTERLIST = \
    "/home/meuleman/work/projects/ENCODE3/" \
    "WM20180608_masterlist_FDR0.01_annotations/" \
    "master_list_stats_WM20180608.txt"

PATH_TO_NMF_LOADINGS = \
    "/home/amuratov/fun/60918/60518_NNDSVD_NC16/" \
    "2018-06-08NC16_NNDSVD_Mixture.csv"

DHS_DF_COLUMNS = [
    'seqname',
    'start',
    'end',
    'DHS_width',
    'summit',
    'total_signal',
    'numsamples',
    'numpeaks',
]

def load_reference_genome(path_to_reference_genome):
    return {
        record.id : record.seq
        for record in SeqIO.parse(path_to_reference_genome, "fasta")
    }


def load_dhs_annotations(path_to_dhs_annotations, columns):
    df = pd.read_csv(path_to_dhs_annotations, sep='\t')[columns]
    return df


def load_nmf_loadings(path_to_nmf_loadings):
    return pd.read_csv(path_to_nmf_loadings, sep='\t')


def get_combined_df(dhs_df, nmf_df, length, mean_signal_thresh):
    # Combine the dhs annotations dataframe with the nmf loadings,
    # filter out rows with sequences that are too short or have
    # too weak of a signal
    df = pd.concat([dhs_df, nmf_df], axis=1, sort=False)

    # Filter out rows w/ dhs_width < length
    df = df[df['DHS_width'] >= length]
    # Filter out rows w/ mean_signal < 0.5
    df = df[df.total_signal.values/df.numsamples.values > mean_signal_thresh]

    return df





REFERENCE_GENOME_DICT = load_reference_genome(PATH_TO_REFERENCE_GENOME)

DHS_ANNOTATIONS_DF = load_dhs_annotations(PATH_TO_DHS_MASTERLIST,
                                          DHS_DF_COLUMNS)

NMF_LOADINGS_DF = load_nmf_loadings(PATH_TO_NMF_LOADINGS)


def get_sequence(chrom, start, end):
    return REFERENCE_GENOME_DICT[chrom][start:end]


def get_sequence_bounds(row, half_length):
    # Get the left and right indices of the sequence
    before = row.summit - row.start
    after = row.end - row.summit
    # Edge cases where the summit is to close to the bounds
    if before < half_length:
        excess = half_length - before
        l = row.start
        r = row.summit + half_length + excess
    elif after < half_length:
        excess = half_length - after
        l = row.summit - half_length - after
        r = row.end
    else:
        l = row.summit - half_length
        r = row.summit + half_length
    return l, r


def bad_nucleotides(seq):
    for nt in seq:
        if nt not in ["A", "T", "G", "C"]:
            return True
    return False

# Get sequences (simultaneously trim long sequences to 200)
def get_sequences_and_trim(df, length):
    print("Getting {} sequences...".format(len(df)))

    seqs = []
    half = length // 2
    for i, (row_i, row) in enumerate(df.iterrows(), 0):

        l, r = get_sequence_bounds(row, half)
        seq = get_sequence(row.seqname, l, r)

        if bad_nucleotides(seq):
            df = df.drop(row_i)
            continue

        seqs.append(seq)

        if i % 100000 == 0:
            print("Sequences Loaded: " + str(i))

    print('Finished getting {} sequences'.format(len(seqs)))
    return seqs, df


# Some constants (hardcoded for now)
LENGTH = 100
MEAN_SIGNAL_THRESHOLD = 0.5
FULL_DATAFRAME = get_combined_df(DHS_ANNOTATIONS_DF,
                                 NMF_LOADINGS_DF,
                                 LENGTH,
                                 MEAN_SIGNAL_THRESHOLD)


seqs, full_df = get_sequences_and_trim(FULL_DATAFRAME, LENGTH)

one_hot_seqs = np.array(list(map(utils.seq_to_one_hot, seqs)))

nmf_vectors = full_df.loc[:, 'C1':'C16'].values.astype(float)

components = nmf_vectors.argmax(axis=1)

print(one_hot_seqs)
print(components)
print(full_df.head())
