import numpy as np
import pandas as pd
from progress.bar import IncrementalBar
from Bio import SeqIO

from utils.constants import (
    COMPONENT,
    COMPONENT_COLUMNS,
    DHS_DATA_COLUMNS,
    DHS_WIDTH,
    END,
    NUMSAMPLES,
    PROPORTION,
    RAW_SEQUENCE,
    SEQNAME,
    START,
    SUMMIT,
    TEST_CHR,
    TOTAL_SIGNAL,
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

        df[COMPONENT] = df[COMPONENT_COLUMNS].idxmax(axis=1)
        df[PROPORTION] = (
            df[COMPONENT_COLUMNS].max(axis=1) / df[COMPONENT_COLUMNS].sum(axis=1)
        )

        self.df = df
        self.output_path = output_path

    def add_sequences_column(self, df, genome, length):
        seqs = []
        bar = IncrementalBar('Sequences', max=len(df))
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
        masks['test'] = self.df[SEQNAME] == TEST_CHR
        masks['validation'] = self.df[SEQNAME] == VALIDATION_CHR
        masks = ~(masks['test'] | masks['validation'])


# def get_sequence_strength_cutoffs(df, num_sequences):
#     component = df[COMPONENT_COLUMNS].idxmax(axis=1)
#     component_val = df[COMPONENT_COLUMNS].max(axis=1)
#     nmf_sum = df[COMPONENT_COLUMNS].sum(axis=1)

#     df['component'] = component
#     df['component_val'] = component_val
#     df['nmf_sum'] = nmf_sum

#     df['proportion'] = df.component_val / df.nmf_sum

#     strongest = df.groupby('component')['proportion'].nlargest(num_sequences)
#     return np.array([strongest[c].min() for c in COMPONENT_COLUMNS])
