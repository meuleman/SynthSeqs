from collections import defaultdict

from Bio import SeqIO
from pathlib import Path


class FIMOScans:
    def __init__(self, tuned_dir):
        self.tuned_dir = tuned_dir

    def aggregate_seqs(self, components, fimo_dir):
        for component in components:
            seq_hist = defaultdict(list)
            tuned_fastas = Path(self.tuned_dir + str(component)).iterdir()

            for f in tuned_fastas: 
                seqs = SeqIO.parse(f, 'fasta')

                for seq in seqs:
                    iteration = int(seq.id)
                    seq.id = f.name 
                    seq_hist[iteration].append(seq)

            for iteration in seq_hist.keys():
                save_path = f'{fimo_dir}{component}/{iteration}.fasta'

                with open(save_path, 'w') as out:
                    SeqIO.write(seq_hist[iteration], out, 'fasta')

