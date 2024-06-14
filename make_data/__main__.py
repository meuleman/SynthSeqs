import argparse
import os

from utils.constants import (
    DATA_DIR,
    MEAN_SIGNAL,
    OUTPUT_DIR,
    PATH_TO_DHS_MASTERLIST,
    PATH_TO_NMF_LOADINGS,
    PATH_TO_REFERENCE_GENOME,
    PATH_TO_BIOSAMPLES,
    SEQUENCE_LENGTH,
)

from make_data.process import DataManager
from make_data.source import (
    Biosamples,
    DHSAnnotations,
    NMFLoadings,
    ReferenceGenome,
)


def init_dirs(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(output_dir + DATA_DIR):
        os.makedirs(output_dir + DATA_DIR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',
                        default=OUTPUT_DIR,
                        type=str,
                        help='The path of the output parent directory')
    parser.add_argument('--ref',
                        default=PATH_TO_REFERENCE_GENOME,
                        type=str,
                        help='The path to the reference genome fasta file')
    parser.add_argument('--dhs',
                        default=PATH_TO_DHS_MASTERLIST,
                        type=str,
                        help='The path to the dhs annotations file')
    parser.add_argument('--nmf',
                        default=PATH_TO_NMF_LOADINGS,
                        type=str,
                        help='The path to the nmf loadings file')
    parser.add_argument('--biosamples',
                        default=PATH_TO_BIOSAMPLES,
                        type=str,
                        help='The path to the biosamples data')
    args = parser.parse_args()

    output_dir = args.output
    init_dirs(output_dir)

    dhs_annotations = DHSAnnotations.from_path(args.dhs)
    nmf_loadings = NMFLoadings.from_path(args.nmf)
    genome = ReferenceGenome.from_path(args.ref)
    biosamples = Biosamples.from_path(args.biosamples)
    # We do a concat in the DataManager so drop the index
    biosamples.data.reset_index(drop=True, inplace=True)

    data_manager = DataManager(dhs_annotations=dhs_annotations,
                               nmf_loadings=nmf_loadings,
                               genome=genome,
                               mean_signal=MEAN_SIGNAL,
                               sequence_length=SEQUENCE_LENGTH,
                               output_path=output_dir + DATA_DIR,
                               biosamples=biosamples)

    data_manager.write_data()


if __name__ == '__main__':
    main()
