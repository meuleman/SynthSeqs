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

from .constants import LOSS, SOFTMAX


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

            raw_seqs = [
                SeqRecord(one_hot_to_seq(x), id=str(j))
                for j, x in zip(range(*vector_id_range), seqs)
            ]

            softmax = nn.Softmax(dim=1)
            softmax_out = softmax(pred)[:, target_class].mean().item()

            if verbose or (i == (iters - 1)) or (i == 0):
                self.save_training_history(save_dir,
                                           i,
                                           raw_seqs,
                                           loss,
                                           softmax_out,
                                           opt_z_norm,
                                           seed_dir,
                                           vector_id_range)
        
    def save_training_history(self,
                              save_dir,
                              iteration,
                              raw_seqs,
                              loss,
                              softmax_out,
                              opt_z_norm,
                              seed_dir,
                              vector_id_range):

        with open(save_dir + str(iteration) + '.fasta', 'a') as f:
            SeqIO.write(raw_seqs, f, 'fasta')

<<<<<<< HEAD
        if vector_id_range:
            tag = '_' + str(vector_id_range[0]) + '_' + str(vector_id_range[1])
        else:
            tag = ''

        with open(save_dir + f'loss/loss{tag}.txt', 'a') as f:
            f.write(str(loss.item()) + '\n')
=======
    def optimize_multiple(self,
                          opt_zs,
                          target_class,
                          iters,
                          save_dir):

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
            print(opt_z_norm.mean(), end='\t')
            print(opt_z_norm.std())
            seq = self.generator(opt_z_norm).transpose(-2, -1).squeeze(1)
 
            # We forward pass up to the fully connected layer
            # before the final softmax operation.
            pred = self.classifier.no_softmax_forward(seq).squeeze()

            loss = -(pred[:, target_class].mean())
            loss.backward()
            self.optimizer.step()

            seqs = seq.cpu().detach().numpy().squeeze().transpose(0, 2, 1)

            raw_seqs = [SeqRecord(one_hot_to_seq(x), id=str(j)) for j, x in enumerate(seqs)]

            softmax = nn.Softmax(dim=1)
            softmax_out = softmax(pred)[:, target_class].mean().item()
        
            with open(save_dir + str(i) + '.fasta', 'w') as f:
                SeqIO.write(raw_seqs, f, 'fasta')

            with open(save_dir + 'loss/loss.txt', 'a') as f:
                f.write(str(loss.item()) + '\n')

            with open(save_dir + 'softmax/softmax.txt', 'a') as f:
                f.write(str(softmax_out) + '\n')
>>>>>>> 5577b00d865ab7dce789e138a89d4feb016acb8f

        with open(save_dir + f'softmax/softmax{tag}.txt', 'a') as f:
            f.write(str(softmax_out) + '\n')

        if seed_dir:
            seed = opt_z_norm.cpu().detach().numpy().squeeze()
            np.save(seed_dir + str(iteration) + f'seed{tag}.npy', seed)

