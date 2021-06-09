from utils.seq import seq_to_one_hot
from Bio import SeqIO

import numpy as np

NT_TO_NUM = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
}

NUM_SEQS = 5000
LEN_SEQS = 200

def tune_seqs_path(comp, it):
    return f'../tuning/640filters/{comp}/{it}.fasta'

def load_in_seqs(c, i):         
    seqs = np.zeros((NUM_SEQS, LEN_SEQS))

    fasta_path = tune_seqs_path(c, i)
    with open(fasta_path, 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            idx = int(record.id)
            number_seq = np.array([NT_TO_NUM[nt] for nt in record.seq])

            seqs[idx] = number_seq
                                            
    return seqs

def calculate_skew(seq_int_repr):
    nts, counts = np.unique(seq_int_repr, return_counts=True)
    at_skew = np.abs(np.log2(counts[0] / counts[3]))
    cg_skew = np.abs(np.log2(counts[1] / counts[2]))
    return (at_skew + cg_skew) / 2

#s = {}
#for c in range(16):
#    skews = []
#    for i in range(5000):
#        seqs = load_in_seqs(c, i)
#        skew = calculate_skew(seqs)
#        skews.append(skew)
#    print(c)
#    s[c] = skews
