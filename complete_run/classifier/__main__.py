from torch import cuda, device

from .analysis import Evaluator
from .models import conv_net
from .trainer import ClassifierTrainer, HyperParameterSearch, ParameterGroup

def main():
    dev = device("cuda" if cuda.is_available() else "cpu")

    #TODO: This is all temporary
    EPOCHS = 20
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
        'filters': [32, 64, 96],
        'pool_size': [5, 10, 2],
        'fully_connected': [25, 50, 100, 150],
        'drop': [0.0, 0.2, 0.5],
    })

    hyper_param_search = HyperParameterSearch(trainer,
                                              model_params_group,
                                              optimizer_params_group)

    hyper_param_search.search()
    hyper_param_search.save(OUTPUT_DIR, 'search_results.csv')


if __name__ == "__main__":
    main()
