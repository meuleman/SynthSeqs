from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from itertools import compress
import argparse
from pathlib import Path

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

    def tune(
        self,
        opt_zs,
        target_component,
        num_iterations,
        save_interval,
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
                raw_seqs = [
                    SeqRecord(one_hot_to_seq(x), id=str(j))
                    for j, x in enumerate(seqs)
                ]
                self.save_sequences(raw_seqs, i, save_dir)

                self.save_performance_metrics(
                    pred,
                    seq,
                    opt_z_norm,
                    save_dir,
                    i,
                )

    def save_sequences(self, seq_records, iteration, save_dir):
        file = save_dir + "sequences/" + f'iteration_{iteration}.fasta'
        with open(file, 'w') as f:
            SeqIO.write(raw_seqs, f, 'fasta')

    def _performance_df_columns(self):
        columns = ["seq_id"]
        # Columns for loss value for all components
        for i in range(TOTAL_CLASSES):
            columns.append(f"loss_{i + 1}")
        # Columns for softmax value for all components
        for i in range(TOTAL_CLASSES):
            columns.append(f"softmax_{i + 1}")
        columns.append("skew")
        return columns

    def _performance_metrics_records(self, loss, softmax, seqs):
        records = []
        for i, seq in enumerate(seqs):
            record = [i]
            # Append loss
            for loss_val in loss[i]:
                record.append(loss_val)
            # Append softmax
            for softmax_val in softmax[i]:
                record.append(softmax_val)

            # TODO: Per sequence skew function
            skew = -1
            record.append(skew)
            records.append(tuple(record))

        return records


    def save_performance_metrics(
        self,
        pred,
        seqs,
        opt_z_norm,
        save_dir,
        iteration,
    ):
        # Convert all items to numpy, cpu
        softmax_operation = nn.Softmax(dim=1)
        softmax_out = softmax_operation(pred).cpu().detach().numpy()

        seqs = seqs.cpu().detach().numpy().squeeze().transpose(0, 2, 1)
        pred = pred.cpu().detach().numpy()
        loss = -1 * pred
        seeds = opt_z_norm.cpu().detach().numpy().squeeze()

        columns = self._performance_df_columns()
        records = self._performance_metrics_records(loss, softmax_out, seqs)
        dataframe = pd.DataFrame.from_records(records, columns=columns)

        file = save_dir + "performance/" + f'iteration_{iteration}.csv'
        dataframe.to_csv(file, sep=',')

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
