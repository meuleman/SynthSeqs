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
    MODEL_DIR,
    TUNING_DIR,
    VECTOR_DIR,
    VECTOR_FILE,
)

from .optimize import SequenceTuner
from .vectors import TuningVectors


def optimize(vector_id_range, target_class, args):
    output_dir = args.output
    name = args.name
    if name[-1] != '/':
        name += '/'
    verbose = args.verbose

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('comp_and_base', 
                        type=int,
                        help='Integer encoding component and vector id information')
    parser.add_argument('-b', '--batch_mode',
                        type=int,
                        default=1,
                        help='0 for 1000 seqs at a time, 1 for 100k at a time')
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

    if args.batch_mode == 1:
        c = comp_and_base // 20 
        range_base = (comp_and_base % 20) * 5000 
        vector_id_range = (range_base, range_base + 5000)
    else:
        c = comp_and_base 
        vector_id_range = (0, 1000)

    optimize(vector_id_range, c, args)

