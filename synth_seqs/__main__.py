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

from synth_seqs.synth_seqs import SequenceTuner


def generate_tuning_vectors(num_vectors, len_vectors, seed=None):
    if seed:
        # This is now the preferred way to generate reproducible pseudo-random numbers without
        # affecting the global random state.
        random_number_generator = np.random.default_rng(seed)
        return random_number_generator.normal(0, 1, (num_vectors, len_vectors))
    else:
        return np.random.normal(0, 1, (num_vectors, len_vectors))


def tune(
    num_sequences,
    target_component,
    random_seed,
    num_iterations,
    save_interval,
    output_dir,
    run_name,
):
    if run_name[-1] != '/':
        run_name += '/'

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

    tuner_params = {
        'lr': 0.017,
        'betas': (0.8, 0.59)
    }

    tuner = SequenceTuner(generator, classifier, tuner_params, dev)

    opt_zs = generate_tuning_vectors(num_sequences, NZ, seed=random_seed)
    save_dir = output_dir + TUNING_DIR + run_name + f'{target_component}/'

    start = time.time()
    tuner.tune(
        opt_zs,
        target_component,
        num_iterations,
        save_interval,
        random_seed,
        save_dir,
    )

    elapsed = time.time() - start


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_sequences',
                        type=int,
                        help='Total number of sequences to tune (int, 1-5000)')
    parser.add_argument('-c', '--component',
                        type=int,
                        help='Regulatory component to tune sequences toward (int, 1-16)')
    parser.add_argument('--seed',
                        type=int,
                        help='Random seed to use for generating fixed random tuning seeds (int)')
    parser.add_argument('-i', '--num_iterations',
                        type=int,
                        default=10000,
                        help='Total number of tuning iterations to tune the sequences for (int)')
    parser.add_argument('--save_interval',
                        type=int,
                        help='Regular tuning iteration interval to save sequences (int)')
    parser.add_argument('-o', '--output_dir',
                        default=OUTPUT_DIR,
                        type=str,
                        help='The path of the output parent directory')
    parser.add_argument('--run_name',
                        default='tuning_data',
                        type=str,
                        help='Name of the tuning write directory')
    args = parser.parse_args()

    tune(
        args.num_sequences,
        args.component,
        args.seed,
        args.num_iterations,
        args.save_interval,
        args.output_dir,
        args.run_name,
    )

