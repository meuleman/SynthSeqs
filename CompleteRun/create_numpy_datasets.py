import numpy as np
import pandas as pd
from Bio import SeqIO
import utils

DATA_DIR = 'data/'

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

COMPONENT_COLUMNS = [
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
    'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
]

COMPONENT_COLUMNS_MAP = {
    c: i for c, i in zip(COMPONENT_COLUMNS, range(16))
}


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
        l = row.summit - half_length - excess
        r = row.end
    else:
        l = row.summit - half_length
        r = row.summit + half_length
    assert r - l == 100
    return l, r


def bad_nucleotides(seq):
    for nt in seq:
        if nt not in ["A", "T", "G", "C"]:
            return True
    return False


# Get sequences (simultaneously trim long sequences to 100)
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


def get_sequence_strength_cutoffs(df, num_sequences):
    component = df[COMPONENT_COLUMNS].idxmax(axis=1)
    component_val = df[COMPONENT_COLUMNS].max(axis=1)
    nmf_sum = df[COMPONENT_COLUMNS].sum(axis=1)

    df['component'] = component
    df['component_val'] = component_val
    df['nmf_sum'] = nmf_sum

    df['proportion'] = df.component_val / df.nmf_sum

    strongest = df.groupby('component')['proportion'].nlargest(num_sequences)

    return {
        c: strongest[c].min() for c in COMPONENT_COLUMNS
    }



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

dset_labels = ['test', 'validation', 'train']

masks = {}
masks['test'] = (full_df.seqname == 'chr1')
masks['validation'] = (full_df.seqname == 'chr2')
masks['train'] = ~(test_mask | validation_mask)


idxs = {
    label: np.where(masks[label])[0]
    for label in dset_labels
}

dfs = {
    label: full_df[masks[label]]
    for label in dset_labels
}

num_seqs_small = {
    'test': 1000,
    'validation': 1000,
    'train': 10000,
}

num_seqs_large = {
    'test': 2000,
    'validation': 2000,
    'train': 20000,
}

cutoffs_small = {
    label: get_sequence_strength_cutoffs(df, num_seqs_small[label])
    for label, df in dfs.items()
}

cutoffs_large = {
    label: get_sequence_strength_cutoffs(df, num_seqs_large[label])
    for label, df in dfs.items()
}

def create_full_mask(nmf_loadings, cutoffs, mask):
    buffer = np.zeros(len(nmf_loadings), dtype=bool)
    for i in range(len(buffer)):
        if mask[i]:
            loading = nmf_loadings[i]
            c = loading.argmax()
            proportion = loading.max() / loading.sum()
            buffer[i] = (proportion > cutoffs[c])
    return buffer

full_masks_small = {
    label: create_full_mask(nmf_loadings, cutoffs_small, masks[label])
    for label in dset_labels
}

full_masks_large = {
    label: create_full_mask(nmf_loadings, cutoffs_large, masks[label])
    for label in dset_labels
}

# print('Saving sequences to {}'.format(DATA_DIR))
# for dset in dset_labels:
#     idx = idxs[dset]
#     print('{0} set: {1} sequences'.format(dset, len(idx)))
#     np.save('data/{}_seqs.npy'.format(dset), one_hot_seqs[idx])
#     np.save('data/{}_components.npy'.format(dset), components[idx])

# full_df.to_csv(DATA_DIR + 'seq_ref.csv')

for label in dset_labels:
    idx = full_masks_small[label]
    print('{0} set: {1} sequences'.format(label, len(idx)))
    np.save('data/{}_seqs_classifier_small.npy'.format(label), one_hot_seqs[idx])
    np.save('data/{}_components_classifier_small.npy'.format(label), components[idx])

for label in dset_labels:
    idx = full_masks_large[label]
    print('{0} set: {1} sequences'.format(label, len(idx)))
    np.save('data/{}_seqs_classifier_large.npy'.format(label), one_hot_seqs[idx])
    np.save('data/{}_components_classifier_large.npy'.format(label), components[idx])
