from Bio import SeqIO
import pandas as pd

from utils.constants import DHS_DATA_COLUMNS, REFERENCE_GENOME_FILETYPE
from utils.exceptions import UnimplementedMethodException


class DataSource:

    def __init__(self, data, filepath):
        self.raw_data = data
        self.filepath = filepath

    @classmethod
    def from_path(cls, path):
        raise UnimplementedMethodException()

    @classmethod
    def from_dict(cls, data_dict):
        raise UnimplementedMethodException()

    @property
    def data(self):
        return self.raw_data


class DHSAnnotations(DataSource):

    @classmethod
    def from_path(cls, path):
        df = pd.read_csv(path, sep='\t')
        return cls(df, path)

    @classmethod
    def from_dict(cls, data_dict):
        df = pd.DataFrame.from_dict(data_dict)
        return cls(df, filepath=None)
    
    @property
    def data(self):
        return self.raw_data[DHS_DATA_COLUMNS]


class NMFLoadings(DataSource):

    @classmethod
    def from_path(cls, path):
        df = pd.read_csv(path, sep='\t', index_col=0)
        return cls(df, path)

    @classmethod
    def from_dict(cls, data_dict):
        df = pd.DataFrame.from_dict(data_dict)
        return cls(df, filepath=None)


class ReferenceGenome(DataSource):

    @classmethod
    def from_path(cls, path):
        genome_dict = {
            record.id : record.seq
            for record in SeqIO.parse(path, REFERENCE_GENOME_FILETYPE)
        }
        return cls(genome_dict, path)

    @classmethod
    def from_dict(cls, data_dict):
        return cls(data_dict, filepath=None)

    @property
    def genome(self):
        return self.data

    def sequence(self, chrom, start, end):
        chrom_sequence = self.genome[chrom]

        assert end < len(chrom_sequence), (
            f"Sequence position bound out of range for chromosome {chrom}. "
            f"{chrom} length {len(chrom_sequence)}, requested position {end}."
        )
        return chrom_sequence[start:end]
