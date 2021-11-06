import numpy as np
import pandas as pd
from progress.bar import IncrementalBar
from Bio import SeqIO

from utils.constants import (
    CLASSIFIER,
    COMPONENT,
    COMPONENTS,
    COMPONENT_COLUMNS,
    csv_data_filename,
    data_filename,
    DHS_DATA_COLUMNS,
    DHS_WIDTH,
    END,
    GENERATOR,
    NUMSAMPLES,
    NUM_SEQS_PER_COMPONENT,
    PROPORTION,
    RAW_SEQUENCE,
    SEQNAME,
    SEQUENCES,
    START,
    SUMMIT,
    TEST,
    TEST_CHR,
    TOTAL_CLASSES,
    TOTAL_SIGNAL,
    TRAIN,
    VALIDATION,
    VALIDATION_CHR,
)
from utils.seq import seq_to_one_hot, bad_nucleotides


COMPONENT_COLUMNS_MAP = {
    c: i for c, i in zip(COMPONENT_COLUMNS, range(TOTAL_CLASSES))
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

        # Preprocess the raw dhs annotation records, pull the remaining sequences
        # from the reference genome and add helpful columns.
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
        """
        Query the reference genome for each DHS and add the raw sequences
        to the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe of DHS annotations and NMF loadings.
        genome : ReferenceGenome(DataSource)
            A reference genome object to query for sequences.
        length : int
            Length of a DHS.
        """
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
        """Calculate the sequence coordinates (bounds) for a given DHS.
        """
        half = length // 2

        if (summit - start) < half:
            return start, start + length
        elif (end - summit) < half:
            return end - length, end

        return summit - half, summit + half

    def write_data(self):
        """Write the generator and classifier numpy datasets.
        """
        # Create masks for the heldout chromosomes and the train set.
        masks = {}
        masks[TEST] = (self.df[SEQNAME] == TEST_CHR)
        masks[VALIDATION] = (self.df[SEQNAME] == VALIDATION_CHR)
        masks[TRAIN] = ~(masks[TEST] | masks[VALIDATION])

        for label in masks.keys():
            df = self.df[masks[label]]

            # Shuffle the rows of the dataframe. This is done to make
            # tiebreaking random for the step of pulling the
            # proportion rank masks.
            df = df.sample(frac=1)

            # Convert sequences to one-hot encodings.
            sequences = df[RAW_SEQUENCE].values
            one_hots = np.array(list(map(seq_to_one_hot, sequences)))
            components = df[COMPONENT].values

            # For each component, rank descending by proportion and make
            # a mask keeping the top N sequences. This subset of sequences
            # will be the classifier data.
            prop_mask = (
                df.groupby(COMPONENT)[PROPORTION]
                  .rank(ascending=False, method='first')
            ) <= NUM_SEQS_PER_COMPONENT[label]
           
            self._write_generator_data(label, one_hots, components)
            self._write_classifier_data(label, one_hots, components, prop_mask)
            self._save_classifier_df(label, df, prop_mask)



    def _write_generator_data(self, label, one_hots, components):
        seq_filename = data_filename(label, SEQUENCES, GENERATOR)
        comp_filename = data_filename(label, COMPONENTS, GENERATOR)
        
        print(f'Writing generator {label} data.')
        np.save(self.output_path + seq_filename, one_hots)
        np.save(self.output_path + comp_filename, components)

    def _write_classifier_data(self, label, one_hots, components, mask):
        seq_filename = data_filename(label, SEQUENCES, CLASSIFIER)
        comp_filename = data_filename(label, COMPONENTS, CLASSIFIER)

        print(f'Writing classifier {label} data.')
        np.save(self.output_path + seq_filename, one_hots[mask])
        np.save(self.output_path + comp_filename, components[mask])

    def _save_classifier_df(self, label, df, mask):
        csv_filename = csv_data_filename(label, "all", CLASSIFIER)
        df[mask].to_csv(self.output_path + csv_filename)
