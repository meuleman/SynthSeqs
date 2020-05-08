import numpy as np
import pandas as pd
from progress.bar import IncrementalBar
from Bio import SeqIO

from utils.constants import (
    COMPONENT,
    COMPONENT_COLUMNS,
    data_filename,
    DHS_DATA_COLUMNS,
    DHS_WIDTH,
    END,
    NUMSAMPLES,
    NUM_SEQS_PER_COMPONENT,
    PROPORTION,
    RAW_SEQUENCE,
    SEQNAME,
    START,
    SUMMIT,
    TEST,
    TEST_CHR,
    TOTAL_SIGNAL,
    TRAIN,
    VALIDATION,
    VALIDATION_CHR,
)
from utils.seq_utils import seq_to_one_hot, bad_nucleotides


COMPONENT_COLUMNS_MAP = {
    c: i for c, i in zip(COMPONENT_COLUMNS, range(16))
}


class DataManager:
    def __init__(self,
                 dhs_annotations,
                 nmf_loadings,
                 genome,
                 mean_signal,
                 sequence_length,
                 output_path):

        assert len(dhs_annotations.data) == len(nmf_loadings.data), (
            f"Number of DHS rows {len(dhs_annotations.data)} not equal to " 
            f"number of NMF rows {len(nmf_loadings.data)}."
        )
        df = pd.concat([dhs_annotations.data, nmf_loadings.data],
                        axis=1,
                        sort=False)
        df = df[df[DHS_WIDTH] >= sequence_length]
        df = df[df[TOTAL_SIGNAL].values/df[NUMSAMPLES].values > mean_signal]

        df = self.add_sequences_column(df, genome, sequence_length)

        df[COMPONENT] = (df[COMPONENT_COLUMNS]
                        .idxmax(axis=1)
                        .apply(lambda x: int(x[1:]) - 1))
        df[PROPORTION] = (
            df[COMPONENT_COLUMNS].max(axis=1) / df[COMPONENT_COLUMNS].sum(axis=1)
        )

        self.df = df
        self.output_path = output_path

    def add_sequences_column(self, df, genome, length):
        seqs = []
        bar = IncrementalBar('Sequences pulled from genome', max=len(df))
        for row_i, row in df.iterrows():
            l, r = self.sequence_bounds(row[SUMMIT],
                                        row[START],
                                        row[END],
                                        length)
            seq = genome.sequence(row[SEQNAME], l, r)
            if bad_nucleotides(seq):
                df = df.drop(row_i)
            else:
                seqs.append(seq)

            bar.next()

        bar.finish()
        df[RAW_SEQUENCE] = seqs
        return df

    def sequence_bounds(self, summit, start, end, length):
        half = length // 2
        if (summit - start) < half:
            return start, start + length
        elif (end - summit) < half:
            return end - length, end
        return summit - half, summit + half

    def write_data(self):
        masks = {}
        masks[TEST] = (self.df[SEQNAME] == TEST_CHR)
        masks[VALIDATION] = (self.df[SEQNAME] == VALIDATION_CHR)
        masks = ~(masks[TEST] | masks[VALIDATION])

        for label in masks.keys():
            df = self.df[masks[label]]

            # Shuffle the rows of the dataframe. This is done to make
            # tiebreaking random for the step of pulling the
            # proportion rank masks.
            df = df.sample(frac=1)

            sequences = df[RAW_SEQUENCE].values
            one_hots = np.array(list(map(seq_to_one_hot, sequences)))
            components = df[COMPONENT].values

            # For each component, rank descending by proportion and make
            # a mask keeping the top N sequences.
            prop_mask = (
                df.groupby(COMPONENT)[PROPORTION]
                  .rank(ascending=False, method='first')
            ) <= NUM_SEQS_PER_COMPONENT[label]
            
            self._write_generator_data(label, one_hots, components)
            self._write_classifier_data(label, one_hots, components, prop_mask)

    def _write_generator_data(self, label, one_hots, components):
        seq_filename = data_filename(label, 'sequences', 'generator')
        comp_filename = data_filename(label, 'components', 'generator')
        
        print(f'Writing generator {label} data.')
        np.save(self.output_path + seq_filename, one_hots)
        np.save(self.output_path + comp_filename, components)

    def _write_classifier_data(self, label, one_hots, components, mask):
        seq_filename = data_filename(label, 'sequences', 'classifier')
        comp_filename = data_filename(label, 'components', 'classifier')

        print(f'Writing classifier {label} data.')
        np.save(self.output_path + seq_filename, one_hots[mask])
        np.save(self.output_path + comp_filename, components[mask])
