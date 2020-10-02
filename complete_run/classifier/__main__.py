import argparse
import os

from torch import cuda, device

from utils.constants import (
    CLASSIFIER_MODEL_FILE,
    DATA_DIR,
    FIGURE_DIR,
    MODEL_DIR,
    OUTPUT_DIR, 
)

from .analysis import Evaluator
from .models import conv_net, conv_net_one_layer
from .trainer import ClassifierTrainer, HyperParameterSearch, ParameterGroup


def classifier_trainer(output_dir):
    dev = device("cuda" if cuda.is_available() else "cpu")

    #TODO: This is all temporary
    EPOCHS = 1000
    BATCH_SIZE = 256
    MODEL = conv_net_one_layer
    DEVICE = dev
    trainer = ClassifierTrainer(EPOCHS,
                                BATCH_SIZE,
                                MODEL,
                                DEVICE,
                                output_dir + DATA_DIR)
    return trainer

def hyperparam_search(output_dir):
    trainer = classifier_trainer(output_dir)

    ### ALL ###
    optimizer_params_group = ParameterGroup({
        'lr': [0.001],
        'betas': [(0.9, 0.99)],
    })
    ### MODEL PARAMS ###
    model_params_group = ParameterGroup({
        'filters': [100, 75, 50, 32, 16, 8],
        'pool_size': [25, 100],
        'fully_connected': [100],
        'drop': [0.3, 0.4, 0.5],
    })

    hyper_param_search = HyperParameterSearch(trainer,
                                              model_params_group,
                                              optimizer_params_group,
                                              plot_dir=output_dir + FIGURE_DIR)

    hyper_param_search.search()
    hyper_param_search.save(output_dir + FIGURE_DIR, 'search_results.csv')

def train_model(output_dir):
    trainer = classifier_trainer(output_dir)

    optimizer_params = {
        'lr': 0.001,
        'betas': (0.9, 0.99)
    }

    model_params = {
        'filters': 100, 
        'pool_size': 200,
        'fully_connected': 100,
        'drop': 0.5,
    }

    trainer.train(model_params, optimizer_params)

    trainer.plot(output_dir + FIGURE_DIR)
    trainer.save(CLASSIFIER_MODEL_FILE, output_dir + MODEL_DIR)

def init_dirs(output_dir):
    assert os.path.exists(output_dir + DATA_DIR), 'Data directory is missing'
     
    if not os.path.exists(output_dir + FIGURE_DIR):
        os.makedirs(output_dir + FIGURE_DIR)

    if not os.path.exists(output_dir + MODEL_DIR):
        os.makedirs(output_dir + MODEL_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',
                        default=OUTPUT_DIR,
                        type=str,
                        help='The path of the output parent directory')
    args = parser.parse_args()

    output_dir = args.output
    init_dirs(output_dir)

    train_model(output_dir) 

