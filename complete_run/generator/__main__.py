from torch import cuda, device

from .constants import NZ
from .models import snp_generator, snp_discriminator
from .trainer import GeneratorTrainer


if __name__ == '__main__':
    dev = device("cuda" if cuda.is_available() else "cpu")

    ITERATIONS = 300000
    BATCH_SIZE = 128
    GENERATOR = snp_generator
    DISCRIMINATOR = snp_discriminator
    DEVICE = dev
    DATA_DIR = '/home/pbromley/synth-seqs-data/'
    trainer = GeneratorTrainer(ITERATIONS,
                               BATCH_SIZE,
                               GENERATOR,
                               DISCRIMINATOR,
                               DEVICE,
                               DATA_DIR)

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
        'num_filters': 320,
        'len_filters': 15,
        'transpose_size': 10,
    }
    discriminator_params = {
        'num_filters': 320,
        'len_filters': 15,
        'pool_size': 20,
        'fully_connected': 100,
    }

    trainer.train(generator_params,
                  discriminator_params,
                  optimizer_params_g,
                  optimizer_params_d)

    FIGURE_DIR = '/home/pbromley/synth-seqs-figures/generator/'
    MODEL_DIR = '/home/pbromley/synth-seqs-models/generator/'

    trainer.plot_seqs(40, 'generated_seqs.png', FIGURE_DIR)
    trainer.save(MODEL_DIR)

