from torch import cuda, device

from .analysis import Evaluator
from .models import conv_net
from .trainer import ClassifierTrainer, HyperParameterSearch, ParameterGroup

def main():
    dev = device("cuda" if cuda.is_available() else "cpu")

    #TODO: This is all temporary
    EPOCHS = 30
    BATCH_SIZE = 256
    MODEL = conv_net
    DEVICE = dev
    DATA_DIR = '/home/pbromley/synth-seqs-data/'
    trainer = ClassifierTrainer(EPOCHS,
                                BATCH_SIZE,
                                MODEL,
                                DEVICE,
                                DATA_DIR)
    OUTPUT_DIR = '/home/pbromley/synth-seqs-figures/'

    ### ALL ###
    optimizer_params_group = ParameterGroup({
        'lr': [0.0018],
        'betas': [(0.9, 0.99)],
    })
    ### MODEL PARAMS ###
    model_params_group = ParameterGroup({
        'filters': [(32, 8),
                    (32, 16),
                    (64, 16),
                    (64, 32),
                    (80, 16),
                    (80, 32),
                    (96, 16),
                    (96, 32)],
        'pool_size': [5],
        'fully_connected': [100],
        'drop': [0.1, 0.2, 0.3, 0.35],
    })

    hyper_param_search = HyperParameterSearch(trainer,
                                              model_params_group,
                                              optimizer_params_group,
                                              plot_dir=OUTPUT_DIR)

    hyper_param_search.search()
    hyper_param_search.save(OUTPUT_DIR, 'search_results.csv')


if __name__ == "__main__":
    main()
