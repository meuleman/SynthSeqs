import argparse
import os
from generator.constants import NZ

from utils.constants import (
    OUTPUT_DIR,
    DATA_DIR,
    MODEL_DIR,
    FIGURE_DIR,
    TUNING_DIR,
    VECTOR_DIR,
    VECTOR_FILE,
)

from optimize.vectors import TuningVectors


def initialize_fixed_vectors(num_vectors, len_vectors, path, seed=None):
    tv = TuningVectors()
    tv.save_fixed(num_vectors, len_vectors, path, seed=seed)


def init_dirs(output_dir):
    assert os.path.exists(output_dir + DATA_DIR), 'Data directory is missing'
    assert os.path.exists(output_dir + MODEL_DIR), 'Model directory is missing'
     
    if not os.path.exists(output_dir + FIGURE_DIR):
        os.makedirs(output_dir + FIGURE_DIR)

    if not os.path.exists(output_dir + TUNING_DIR):
        os.makedirs(output_dir + TUNING_DIR)

    vector_path = output_dir + VECTOR_DIR
    if not os.path.exists(vector_path):
        os.makedirs(vector_path)
        initialize_fixed_vectors(100000, NZ, vector_path + VECTOR_FILE)


def setup_tuning_dir(output_dir, name):
    dir_name = output_dir + TUNING_DIR + name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for c in range(16):
        comp_dir_name = dir_name + f'{c}/'
        if not os.path.exists(comp_dir_name):
            os.makedirs(comp_dir_name)
        for label in ['loss/', 'softmax/', 'seed/', 'skew/']:
            if not os.path.exists(comp_dir_name + label):
                os.makedirs(comp_dir_name + label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',
                        default=OUTPUT_DIR,
                        type=str,
                        help='The path of the output parent directory')
    parser.add_argument('-n', '--name',
                        default='tuning_data',
                        type=str,
                        help='Name of the tuning write directory')
    args = parser.parse_args()

    output_dir = args.output
    name = args.name
    if name[-1] != '/':
        name += '/'

    init_dirs(output_dir)
    setup_tuning_dir(output_dir, name)

