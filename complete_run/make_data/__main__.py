from utils.constants import MEAN_SIGNAL, SEQUENCE_LENGTH

from .create_numpy_datasets import DataManager
from .data_source import (
    DHSAnnotations,
    NMFLoadings,
    ReferenceGenome,
)

# TODO: Implement CLI, stop hardcoding these.
PATH_TO_REFERENCE_GENOME = \
    "/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa"

PATH_TO_DHS_MASTERLIST = \
    "/home/meuleman/work/projects/ENCODE3/" \
    "WM20180608_masterlist_FDR0.01_annotations/" \
    "master_list_stats_WM20180608.txt"

PATH_TO_NMF_LOADINGS = \
    "/home/amuratov/fun/60918/60518_NNDSVD_NC16/" \
    "2018-06-08NC16_NNDSVD_Mixture.csv"

DATA_ROOT = '/home/pbromley/synth-seqs-data/'


def main():
    dhs_annotations = DHSAnnotations.from_path(PATH_TO_DHS_MASTERLIST)
    nmf_loadings = NMFLoadings.from_path(PATH_TO_NMF_LOADINGS)
    genome = ReferenceGenome.from_path(PATH_TO_REFERENCE_GENOME)

    data_manager = DataManager(dhs_annotations=dhs_annotations,
                               nmf_loadings=nmf_loadings,
                               genome=genome,
                               mean_signal=MEAN_SIGNAL,
                               sequence_length=SEQUENCE_LENGTH,
                               output_path=DATA_ROOT)

    data_manager.write_data()


if __name__ == '__main__':
    main()
