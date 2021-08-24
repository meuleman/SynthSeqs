from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np
from itertools import compress
import argparse
import pandas as pd
from pathlib import Path
import torch
from torch.optim import Adam
import torch.nn as nn

from utils.constants import TOTAL_CLASSES
from utils.seq import one_hot_to_seq

from optimize.constants import (
    LOSS,
    SOFTMAX,
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

    def zero_grads(self):
        self.generator.zero_grad()
        self.classifier.zero_grad()
        self.optimizer.zero_grad()

    @staticmethod
    def _tag(seq_id, component, iteration, seed_num):
        return f"{seed_num}_comp{component}_{iteration}_{seq_id}"

    def tune(
        self,
        opt_zs,
        target_component,
        num_iterations,
        save_interval,
        random_seed,
        save_dir,
    ):
        # Convert the target component to a 0-based index for tuning
        target_component_index = target_component - 1
        assert target_component_index >= 0, "Target component must be btw 1-16 (inclusive)"

        opt_zs = (torch.from_numpy(opt_zs)
                       .float()
                       .to(self.device)
                       .requires_grad_())

        self.optimizer = Adam([opt_zs], **self.optimizer_params)
        h = opt_zs.register_hook(
            lambda grad: grad + torch.zeros_like(grad).normal_(0, 1e-4)
        )

        for i in range(num_iterations):
            self.zero_grads()

            # Normalize seed to be unit normal
            opt_z_norm = (opt_zs - opt_zs.mean()) / opt_zs.std()
            seq = self.generator(opt_z_norm).transpose(-2, -1).squeeze(1)
 
            # We forward pass up to the fully connected layer
            # before the final softmax operation.
            pred = self.classifier.no_softmax_forward(seq).squeeze()

            loss = -(pred[:, target_component_index].mean())
            loss.backward()
            self.optimizer.step()
                
            seqs = seq.cpu().detach().numpy().squeeze().transpose(0, 2, 1)

            # TODO: Add logic for stopping criteria
            if i % save_interval == 0:
                tags = [
                    self._tag(seq_id, target_component, i, random_seed)
                    for seq_id in range(len(seqs))
                ]
                raw_seqs = [
                    SeqRecord(one_hot_to_seq(x), id=tags[j])
                    for j, x in enumerate(seqs)
                ]
                self.save_sequences(raw_seqs, i, save_dir)
                self.save_input_seeds(opt_z_norm, i, save_dir)
                self.save_performance_metrics(
                    pred,
                    seqs,
                    opt_z_norm,
                    tags,
                    i,
                    save_dir,
                )

    # @staticmethod
    # def save_early(input_seeds, random_seed)

    @staticmethod
    def save_input_seeds(seeds, iteration, save_dir):
        input_seeds = seeds.cpu().detach().numpy().squeeze()
        file = save_dir + "input_seeds/" + f"iteration{iteration}.fasta"
        np.save(file, input_seeds)

    @staticmethod
    def save_sequences(seq_records, iteration, save_dir, stop=False):
        if stop:
            file = save_dir + "sequences/" + f'iteration{iteration}_stop.fasta'
        else:
            file = save_dir + "sequences/" + f'iteration{iteration}.fasta'

        with open(file, 'w') as f:
            SeqIO.write(seq_records, f, 'fasta')

    @staticmethod
    def calculate_skew(seq):
        seq_int_repr = seq.argmax(axis=1)
        nts, counts = np.unique(seq_int_repr, return_counts=True)
        at_skew = np.abs(np.log2(counts[0] / counts[3]))
        cg_skew = np.abs(np.log2(counts[1] / counts[2]))
        return (at_skew + cg_skew) / 2

    @staticmethod
    def _performance_df_columns():
        columns = ["tag", "seq_id"]
        # Columns for loss value for all components
        for i in range(TOTAL_CLASSES):
            columns.append(f"loss_{i + 1}")
        # Columns for softmax value for all components
        for i in range(TOTAL_CLASSES):
            columns.append(f"softmax_{i + 1}")
        columns.append("skew")
        return columns

    def _performance_metrics_records(self, tags, loss, softmax, seqs):
        records = []
        for i, seq in enumerate(seqs):
            record = [tags[i], i]
            # Append loss
            for loss_val in loss[i]:
                record.append(loss_val)
            # Append softmax
            for softmax_val in softmax[i]:
                record.append(softmax_val)

            skew = self.calculate_skew(seq)
            record.append(skew)
            records.append(tuple(record))

        return records

    def save_performance_metrics(
        self,
        pred,
        seqs,
        opt_z_norm,
        tags,
        iteration,
        save_dir,
    ):
        # Convert all items to numpy, cpu
        softmax_operation = nn.Softmax(dim=1)
        softmax_out = softmax_operation(pred).cpu().detach().numpy()

        pred = pred.cpu().detach().numpy()
        loss = -1 * pred

        columns = self._performance_df_columns()
        records = self._performance_metrics_records(tags, loss, softmax_out, seqs)
        dataframe = pd.DataFrame.from_records(records, columns=columns)

        file = save_dir + "performance/" + f'iteration_{iteration}.csv'
        dataframe.to_csv(file, sep=',', index=False)

        # with open(save_dir + f'loss/loss{tag}.txt', 'a') as f:
        #     f.write(str(loss.item()) + '\n')
        #
        # np.save(save_dir + f'softmax/softmax{tag}.npy', softmax_out)
        #
        # seed = opt_z_norm.cpu().detach().numpy().squeeze()
        # np.save(save_dir + f'seed/seed{tag}.npy', seed)
        #
        # self.save_skew(seqs, save_dir, tag)

    # def save_skew(self, seqs, save_dir, tag):
    #     seq_int_repr = seqs.argmax(axis=2)
    #     skew = self.calculate_skew(seq_int_repr)
    #     with open(save_dir + f'skew/skew{tag}.txt', 'a') as f:
    #         f.write(str(skew) + '\n')
    #
    # def calculate_skew(self, seq_int_repr):
    #     nts, counts = np.unique(seq_int_repr, return_counts=True)
    #     at_skew = np.abs(np.log2(counts[0] / counts[3]))
    #     cg_skew = np.abs(np.log2(counts[1] / counts[2]))
    #     return (at_skew + cg_skew) / 2
