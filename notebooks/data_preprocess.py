import numpy as np
import pandas as pd
from Bio import SeqIO

import utils

def load_chr_seq(filename):
    chr_dict = {record.id:record.seq for record in SeqIO.parse(filename, "fasta")}
    return chr_dict

def get_sequence(chrom, start, end):
    return CHR_SEQ_DICT[chrom][start:end+1]

# Get sequences (simultaneously trim long sequences to 200)
def get_sequences_and_trim(df, length):
    print("Getting " + str(len(df)) + " sequences...")
    seqs = [None]*len(df)
    l, r = 0, 0
    half = length // 2
    for i, (row_i, row) in enumerate(df.iterrows(), 0):
        before = row.summit - row.start
        after = row.end - row.summit
        if before < half:
            l = row.start
            r = row.summit + ((length-1) - before)
        elif after < half:
            l = row.summit - ((length-1) - after)
            r = row.end
        else:
            l = row.summit - half
            r = row.summit + (half-1)
        seq = get_sequence(row.seqname, l, r)
        
        bad = False
        for nt in seq:
            if nt not in ["A", "T", "G", "C"]:
                bad = True
                break
        if bad:
            df = df.drop(row_i)
            continue
                
        seqs[i] = seq
        if i % 100000 == 0:
            print("Sequences Loaded: " + str(i))
    seqs = list(filter(lambda x: x != None, seqs))
    return seqs, df

def get_rcs(seqs):
    return [seq.reverse_complement() for seq in seqs]

def one_hot_fixed_len(seqs, length):
    one_hot = np.zeros((len(seqs), length, 4))
    for i, seq in enumerate(seqs):
        one_hot[i] = utils.seq_to_one_hot(seq)
        if i % 100000 == 0:
            print("Sequences Converted: " + str(i))
    return one_hot

if __name__ == '__main__':
    length = 100
    mean_sig_cut = 0.5
    n_per_class = 20000
    train_pct = 0.8
    seed = False
    strong = False
    data_path = "/home/pbromley/generative_dhs/data_numpy/"

    print("**** Begin Process ****")
    print("Loading Reference Genome... ", end="")
    # get sequences from reference genome
    path_to_ref_genome = "/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa"
    CHR_SEQ_DICT = load_chr_seq(path_to_ref_genome)         # dictionary of all raw sequences
    print("Done")
    print("Loading NMF DataFrames... ", end="")
    annot_df = pd.read_csv('/home/meuleman/work/projects/ENCODE3/WM20180608_masterlist_FDR0.01_annotations/master_list_stats_WM20180608.txt', sep="\t")
    annot_df = annot_df[['seqname', 'start', 'end', 'DHS_width', 'summit', 'total_signal', 'numsamples', 'numpeaks']]
    nmf_vectors_df = pd.read_csv('/home/amuratov/fun/60918/60518_NNDSVD_NC16/2018-06-08NC16_NNDSVD_Mixture.csv', sep="\t")
    nmf_vectors_df = nmf_vectors_df.drop('Unnamed: 0', axis=1)
    df = pd.concat([annot_df, nmf_vectors_df], axis=1, sort=False)
    print("Done")

    df = df[df['DHS_width'] >= length]   # Filter out rows w/ dhs width < length
    print("Filtered out sequences less than length " + str(length))
    df = df[(df.total_signal.values/df.numsamples.values) > mean_sig_cut]  
    print("Filtered out sequences with mean signal less than " + str(mean_sig_cut))

    seqs, df = get_sequences_and_trim(df, length)
    print("Getting Reverse Complements...", end=" ")
    rc_seqs = get_rcs(seqs)
    print("Done")
    print("Getting NMF Loadings...", end=" ")
    nmf_vectors = df.loc[:, 'C1':'C16'].values.astype(float)
    print("Done")
    print("Converting sequences to one-hot encodings of dim (" + str(length) + ", 4)")
    one_hot_seqs = one_hot_fixed_len(seqs, length)
    one_hot_rc_seqs = one_hot_fixed_len(rc_seqs, length)


    if strong:
        print("Getting top " + str(n_per_class) + " sequences per class, ranked by highest proportion of NMF loading taken up by majority component... ", end="")
        idx_dict = {}
        num_components = nmf_vectors.shape[1]
        for i in range(num_components):     
            where = np.where(nmf_vectors.argmax(axis=1) == i)[0]
            where_idxs = np.argsort(nmf_vectors[where, i]/nmf_vectors[where].sum(axis=1))[::-1]
            idx_dict[i] = where[where_idxs][:n_per_class]
    
        strong_one_hot = np.zeros((n_per_class*num_components, length, 4))
        strong_one_hot_rc = np.zeros((n_per_class*num_components, length, 4))
        strong_nmfs = np.zeros((n_per_class*num_components, num_components))
        strong_signals = np.zeros((n_per_class*num_components, 2))

        total_signal = df.total_signal.values
        mean_signal = df.total_signal.values / df.numsamples.values

        place = 0
        for i in range(num_components):
            strong_one_hot[place:place+n_per_class] = one_hot_seqs[idx_dict[i]]
            strong_one_hot_rc[place:place+n_per_class] = one_hot_rc_seqs[idx_dict[i]]
            strong_nmfs[place:place+n_per_class] = nmf_vectors[idx_dict[i]]
            strong_signals[place:place+n_per_class][:, 0] = mean_signal[idx_dict[i]]
            strong_signals[place:place+n_per_class][:, 1] = total_signal[idx_dict[i]]
            place += n_per_class
        strong_components = strong_nmfs.argmax(axis=1)
        print("Done")

        print("Getting Test/Train Splits (Test: {0}, Train: {1})... ".format(round(train_pct, 1), round(1-train_pct, 1)), end="")
        if seed:
            np.random.seed(0)
        idx = np.random.permutation(len(strong_one_hot))
        split = np.floor(n_per_class*num_components*train_pct).astype(int)
        one_hot_seqs_train = strong_one_hot[idx[:split]]
        one_hot_rc_seqs_train = strong_one_hot_rc[idx[:split]]
        components_train = strong_components[idx[:split]]
        one_hot_seqs_test = strong_one_hot[idx[split:]]
        one_hot_rc_seqs_test = strong_one_hot_rc[idx[split:]]
        components_test = strong_components[idx[split:]]
        nmf_train = strong_nmfs[idx[:split]]
        nmf_test = strong_nmfs[idx[split:]]
        signals_train = strong_signals[idx[:split]]
        signals_test = strong_signals[idx[split:]]
        print("Done")

        print("Saving data to " + data_path + "... ", end=" ")
        np.save("/home/pbromley/generative_dhs/data_numpy/strong_one_hot_train.npy", one_hot_seqs_train)
        np.save("/home/pbromley/generative_dhs/data_numpy/strong_one_hot_train_rc.npy", one_hot_rc_seqs_train)
        np.save("/home/pbromley/generative_dhs/data_numpy/strong_components_train.npy", components_train)
        np.save("/home/pbromley/generative_dhs/data_numpy/strong_one_hot_test.npy", one_hot_seqs_test)
        np.save("/home/pbromley/generative_dhs/data_numpy/strong_one_hot_test_rc.npy", one_hot_rc_seqs_test)
        np.save("/home/pbromley/generative_dhs/data_numpy/strong_components_test.npy", components_test)
        np.save("/home/pbromley/generative_dhs/data_numpy/strong_nmfs_train.npy", nmf_train)
        np.save("/home/pbromley/generative_dhs/data_numpy/strong_nmfs_test.npy", nmf_test)
        np.save("/home/pbromley/generative_dhs/data_numpy/strong_sig_train.npy", signals_train)
        np.save("/home/pbromley/generative_dhs/data_numpy/strong_sig_test.npy", signals_test)
        print("Done")

        print("Loaded and saved {0} one-hot sequences ({1} train, {2} test) with length {3} and mean signal > {4}".format(
          len(strong_one_hot), len(one_hot_seqs_train), len(one_hot_seqs_test), length, mean_sig_cut))
        print("Top " + str(n_per_class) + " components taken for each class ranked by highest proportion of nmf loading.")
        print("Minimum proportions for each class: ")
        for i in range(num_components):
            print(str(i+1) + ": " + str((strong_nmfs[strong_nmfs.argmax(1) == i].max(1)/strong_nmfs[strong_nmfs.argmax(1) == i].sum(1)).min()))
    else:
        print("Getting Test/Train Splits (Test: {0}, Train: {1})... ".format(round(train_pct, 1), round(1-train_pct, 1)), end="")
        idx = np.random.permutation(len(one_hot_seqs))
        split = np.floor(train_pct*len(one_hot_seqs)).astype(int)
        one_hot_seqs_train = one_hot_seqs[idx[:split]]
        one_hot_seqs_test = one_hot_seqs[idx[split:]]
        one_hot_rc_seqs_train = one_hot_rc_seqs[idx[:split]]
        one_hot_rc_seqs_test = one_hot_rc_seqs[idx[split:]]
        print("Done")


        print("Saving data to " + data_path + "... ", end=" ")
        np.save("/home/pbromley/generative_dhs/data_numpy/ms_one_hot_train.npy", one_hot_seqs_train)
        np.save("/home/pbromley/generative_dhs/data_numpy/ms_one_hot_train_rc.npy", one_hot_rc_seqs_train)
        np.save("/home/pbromley/generative_dhs/data_numpy/ms_one_hot_test.npy", one_hot_seqs_test)
        np.save("/home/pbromley/generative_dhs/data_numpy/ms_one_hot_test_rc.npy", one_hot_rc_seqs_test)
        print("Done")


	
    print("**** Loading Complete ****")
