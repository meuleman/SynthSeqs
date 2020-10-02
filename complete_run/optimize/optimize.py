from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from itertools import compress
import argparse
from pathlib import Path

from utils.seq import one_hot_to_seq

from .constants import (
    LOSS,
    SOFTMAX,
    STOPPING_ITERATIONS,
    WRITE_ITERATIONS,
)


class SequenceTuner:
    def __init__(self,
                 generator,
                 classifier,
                 optimizer_params,
                 device):
        self.generator = generator
        self.classifier = classifier
        self.optimizer_params = optimizer_params
        self.device = device
        torch.manual_seed(0)

    def zero_grads(self):
        self.generator.zero_grad()
        self.classifier.zero_grad()
        self.optimizer.zero_grad()

    def tune(self,
             opt_zs,
             target_class,
             iters,
             save_dir,
             verbose=True,
             seed_dir=None,
             save_skew=False,
             vector_id_range=None):

        opt_zs = (torch.from_numpy(opt_zs)
                       .float()
                       .to(self.device)
                       .requires_grad_())

        self.optimizer = Adam([opt_zs], **self.optimizer_params)
        h = opt_zs.register_hook(
            lambda grad: grad + torch.zeros_like(grad).normal_(0, 1e-4)
        )

        for i in range(iters):
            self.zero_grads()

            # Normalize seed to be unit normal
            opt_z_norm = (opt_zs - opt_zs.mean()) / opt_zs.std()
            seq = self.generator(opt_z_norm).transpose(-2, -1).squeeze(1)
 
            # We forward pass up to the fully connected layer
            # before the final softmax operation.
            pred = self.classifier.no_softmax_forward(seq).squeeze()

            loss = -(pred[:, target_class].mean())
            loss.backward()
            self.optimizer.step()
                
            seqs = seq.cpu().detach().numpy().squeeze().transpose(0, 2, 1)

            if verbose or (i == STOPPING_ITERATIONS[target_class]) or (i in WRITE_ITERATIONS):
                raw_seqs = [
                    SeqRecord(one_hot_to_seq(x), id=str(j))
                    for j, x in zip(range(*vector_id_range), seqs)
                ]

                softmax = nn.Softmax(dim=1)
                softmax_out = softmax(pred)[:, target_class].cpu().detach().numpy()

                self.save_training_history(save_dir,
                                           i,
                                           seqs,
                                           raw_seqs,
                                           loss,
                                           softmax_out,
                                           opt_z_norm,
                                           seed_dir,
                                           vector_id_range)
        
    def save_training_history(self,
                              save_dir,
                              iteration,
                              seqs,
                              raw_seqs,
                              loss,
                              softmax_out,
                              opt_z_norm,
                              seed_dir,
                              vector_id_range):

        with open(save_dir + str(iteration) + '.fasta', 'a') as f:
            SeqIO.write(raw_seqs, f, 'fasta')

        if vector_id_range:
            tag = f'_{vector_id_range[0]}_{vector_id_range[1]}_iter{iteration}'
        else:
            tag = ''

        with open(save_dir + f'loss/loss{tag}.txt', 'a') as f:
            f.write(str(loss.item()) + '\n')

        np.save(save_dir + f'softmax/softmax{tag}.npy', softmax_out)

        if seed_dir:
            seed = opt_z_norm.cpu().detach().numpy().squeeze()
            np.save(seed_dir + f'seed{tag}.npy', seed)

        self.save_skew(seqs, save_dir, tag)

    def save_skew(self, seqs, save_dir, tag):
        seq_int_repr = seqs.argmax(axis=2) 
        skew = self.calculate_skew(seq_int_repr)
        with open(save_dir + f'skew/skew{tag}.txt', 'a') as f:
            f.write(str(skew) + '\n')

    def calculate_skew(self, seq_int_repr):
        nts, counts = np.unique(seq_int_repr, return_counts=True)
        at_skew = np.abs(np.log2(counts[0] / counts[3]))
        cg_skew = np.abs(np.log2(counts[1] / counts[2]))
        return (at_skew + cg_skew) / 2
