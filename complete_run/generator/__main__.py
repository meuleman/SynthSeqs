import argparse
import os

from torch import cuda, device

from utils.constants import (
    DATA_DIR,
    FIGURE_DIR,
    MODEL_DIR,
    OUTPUT_DIR, 
)

from .constants import NZ

from .models import snp_generator, snp_discriminator
from .trainer import GeneratorTrainer


def init_dirs(output_dir):
    assert os.path.exists(output_dir + DATA_DIR), 'Data directory is missing'
     
    if not os.path.exists(output_dir + FIGURE_DIR):
        os.makedirs(output_dir + FIGURE_DIR)

    if not os.path.exists(output_dir + MODEL_DIR):
        os.makedirs(output_dir + MODEL_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',
                        default=OUTPUT_DIR,
                        type=str,
                        help='The path of the output parent directory')
    args = parser.parse_args()

    output_dir = args.output
    init_dirs(output_dir)

    dev = device("cuda" if cuda.is_available() else "cpu")

    ITERATIONS = 300000
    BATCH_SIZE = 128
    GENERATOR = snp_generator
    DISCRIMINATOR = snp_discriminator
    DEVICE = dev
    trainer = GeneratorTrainer(ITERATIONS,
                               BATCH_SIZE,
                               GENERATOR,
                               DISCRIMINATOR,
                               DEVICE,
                               output_dir + DATA_DIR)

    optimizer_params_g = {
        'lr': 0.0002,
        'betas': (0.5, 0.99),
    }
    optimizer_params_d = {
        'lr': 0.0002,
        'betas': (0.5, 0.99),
    }

    generator_params = {
        'nz': NZ,
        'num_filters': 640,
        'len_filters': 15,
        'transpose_size': 10,
    }
    discriminator_params = {
        'num_filters': 640,
        'len_filters': 15,
        'pool_size': 200,
        'fully_connected': 100,
    }

    trainer.train(generator_params,
                  discriminator_params,
                  optimizer_params_g,
                  optimizer_params_d)

    trainer.plot_seqs(40, 'generated_seqs.png', output_dir + FIGURE_DIR)
    trainer.save(output_dir + MODEL_DIR)

