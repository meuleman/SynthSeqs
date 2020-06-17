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


    def save_training_history(self, save_dir, vector_id):
        for label in [LOSS, SOFTMAX]:
            save_path = f'{save_dir}{label}/{vector_id}.txt'
            values = self.training_history[label]

            with open(save_path, 'w') as f:
                for value in values:
                    f.write(str(value) + '\n')

