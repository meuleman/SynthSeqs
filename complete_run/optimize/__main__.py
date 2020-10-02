import argparse
import os
import sys
import time
import numpy as np
import torch
from torch import device, cuda
from torch.optim import Adam

from generator.constants import NZ
from generator.models import snp_generator
from classifier.models import conv_net_one_layer
from utils.constants import (
    CLASSIFIER_MODEL_FILE,
    DISCRIMINATOR_MODEL_FILE,
    GENERATOR_MODEL_FILE,
    OUTPUT_DIR,
    DATA_DIR,
    MODEL_DIR,
    FIGURE_DIR,
    TUNING_DIR,
    VECTOR_DIR,
    VECTOR_FILE,
)

from .optimize import SequenceTuner
from .vectors import TuningVectors


def init_dirs(output_dir):
    assert os.path.exists(output_dir + DATA_DIR), 'Data directory is missing'
    assert os.path.exists(output_dir + MODEL_DIR), 'Model directory is missing'
     
    if not os.path.exists(output_dir + FIGURE_DIR):
        os.makedirs(output_dir + FIGURE_DIR)

    if not os.path.exists(output_dir + TUNING_DIR):
        os.makedirs(output_dir + TUNING_DIR)

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

def optimize(vector_id_range, target_class, args):

    output_dir = args.output
    name = args.name
    if name[-1] != '/':
        name += '/'
    verbose = args.verbose

    init_dirs(output_dir)
    setup_tuning_dir(output_dir, name)

    dev = device("cuda" if cuda.is_available() else "cpu")

    generator_params = {
        'nz': NZ,
        'num_filters': 640,
        'len_filters': 15,
        'transpose_size': 10,
    }
    generator = snp_generator(**generator_params).to(dev)
    generator_path = output_dir + MODEL_DIR + GENERATOR_MODEL_FILE
    generator.load_state_dict(torch.load(generator_path, map_location=dev))
    generator.train(False)

    model_params = {           
        'filters': 100,
        'pool_size': 200,
        'fully_connected': 100,
        'drop': 0.5,
    }
    classifier = conv_net_one_layer(**model_params).to(dev)
    model_path = output_dir + MODEL_DIR + CLASSIFIER_MODEL_FILE
    classifier.load_state_dict(torch.load(model_path, map_location=dev))
    classifier.train(False)

    classifier.eval()
    generator.eval()

    optimizer_params = {
        'lr': 0.017,
        'betas': (0.8, 0.59)
    }

    tuner = SequenceTuner(generator,
                          classifier,
                          optimizer_params,
                          dev)

    assert os.path.exists(output_dir + VECTOR_DIR), \
           f'Create fixed tuning seeds and save them to {output_dir + VECTOR_DIR + VECTOR_FILE}'

    vector_path = output_dir + VECTOR_DIR + VECTOR_FILE 
    vectors = TuningVectors()

    opt_zs = vectors.load_fixed(vector_path, slice(*vector_id_range))
    iters = 10000
    save_dir = output_dir + TUNING_DIR + name + f'{target_class}/'

    start = time.time()
    tuner.tune(opt_zs,
               target_class,
               iters,
               save_dir,
               verbose,
               vector_id_range=vector_id_range)
                   
    elapsed = time.time() - start

def initialize_fixed_vectors(num_vectors, len_vectors, path, seed=None):
    tv = TuningVectors()
    tv.save_fixed(num_vectors, len_vectors, path, seed=seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('comp_and_base', 
                        type=int,
                        help='Integer encoding component and vector id information')
    parser.add_argument('-o', '--output',
                        default=OUTPUT_DIR,
                        type=str,
                        help='The path of the output parent directory')
    parser.add_argument('-v', '--verbose',
                        default=0,
                        type=int,
                        help='Save data at every iteration in verbose mode (0 for non-verbose, 1 for verbose)')
    parser.add_argument('-n', '--name',
                        default='tuning_data',
                        type=str,
                        help='Name of the tuning write directory')
    args = parser.parse_args()

    comp_and_base = args.comp_and_base

    if args.verbose == 0:
        c = comp_and_base // 20 
        range_base = (comp_and_base % 20) * 5000 
        vector_id_range = (range_base, range_base + 5000)
    else:
        c = comp_and_base 
        vector_id_range = (0, 1000)

    optimize(vector_id_range, c, args)

