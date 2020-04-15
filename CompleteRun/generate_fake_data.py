import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

# Creates data with approx. 80% G/C content in the first and final 3rd, 20%
# G/C content in the middle third.
MORE = 0.4
LESS = 0.1
N_SEQS = 1000000
PATH = '/home/pbromley/SynthSeqs/CompleteRun/'

NUCLEOTIDE_MAP = {
    3: 'G',
    2: 'C',
    0: 'A',
    1: 'T',
}

fake_data_left = np.random.choice(4, (N_SEQS, 33), p=[LESS, LESS, MORE, MORE])
fake_data_right = np.random.choice(4, (N_SEQS, 33), p=[LESS, LESS, MORE, MORE])
fake_data_mid = np.random.choice(4, (N_SEQS, 34), p=[MORE, MORE, LESS, LESS])

full_fake_data = np.concatenate(
    [fake_data_left, fake_data_mid, fake_data_right],
    axis=1,
)

freqs = {
    NUCLEOTIDE_MAP[n]: (full_fake_data == n).sum(axis=0) / N_SEQS for n in range(4)
}

plt.figure(figsize=(15, 10))
for n in NUCLEOTIDE_MAP.values():
    plt.plot(range(100), freqs[n], label=n)
plt.legend()
plt.savefig(PATH + 'fake_data_experiment/fake_data_composition.png')

def seq_to_one_hot(seq):
    x = np.zeros((len(seq), 4))
    for (i, bp) in enumerate(seq):
        x[i, bp] = 1
    return x

fake_data_one_hot = np.array([seq_to_one_hot(x) for x in full_fake_data])

np.save(PATH + 'fake_data_experiment/fake_data.npy', fake_data_one_hot)
