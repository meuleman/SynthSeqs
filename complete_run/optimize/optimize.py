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

        self.training_history = {
            LOSS: [],
            SOFTMAX: [],
        }

    def zero_grads(self):
        self.generator.zero_grad()
        self.classifier.zero_grad()
        self.optimizer.zero_grad()

    def collect_training_history(self, pred, loss, target_class):
        softmax = nn.Softmax(dim=0)
        softmax_out = softmax(pred)[target_class].item()
        self.training_history[SOFTMAX].append(softmax_out)
        self.training_history[LOSS].append(loss.item())

    def optimize(self,
                 opt_z,
                 target_class,
                 iters,
                 save_dir,
                 vector_id,
                 collect_train_hist=False):
        opt_z = (torch.from_numpy(opt_z)
                      .float()
                      .to(self.device)
                      .requires_grad_())

        self.optimizer = Adam([opt_z], **self.optimizer_params)
        h = opt_z.register_hook(
            lambda grad: grad + torch.zeros_like(grad).normal_(0, 1e-4)
        )

        raw_seqs = []
        for i in range(iters):
            self.zero_grads()
            seq = self.generator(opt_z).transpose(-2, -1).squeeze(1)

            # We forward pass up to the fully connected layer
            # before the final softmax operation.
            pred = self.classifier.no_softmax_forward(seq).squeeze()

            loss = -(pred[target_class])
            loss.backward()
            self.optimizer.step()

            raw_seq = one_hot_to_seq(seq.cpu().detach().numpy().squeeze().transpose())
            raw_seqs.append(SeqRecord(raw_seq, id=str(i)))

            if collect_train_hist:
                self.collect_training_history(pred, loss, target_class)

        with open(save_dir + str(vector_id) + '.fasta', 'w') as f:
            SeqIO.write(raw_seqs, f, 'fasta')

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

    def save_training_history(self, save_dir, vector_id):
        for label in [LOSS, SOFTMAX]:
            save_path = f'{save_dir}{label}/{vector_id}.txt'
            values = self.training_history[label]

            with open(save_path, 'w') as f:
                for value in values:
                    f.write(str(value) + '\n')

