from torch import cuda, device

from .analysis import Evaluator
from .models import conv_net, conv_net_one_layer
from .trainer import ClassifierTrainer, HyperParameterSearch, ParameterGroup


def classifier_trainer():
    dev = device("cuda" if cuda.is_available() else "cpu")

    #TODO: This is all temporary
    EPOCHS = 1000 
    BATCH_SIZE = 256
    MODEL = conv_net_one_layer
    DEVICE = dev
    DATA_DIR = '/home/pbromley/synth-seqs-data-len-200/'
    trainer = ClassifierTrainer(EPOCHS,
                                BATCH_SIZE,
                                MODEL,
                                DEVICE,
                                DATA_DIR)
    return trainer

def hyperparam_search():
    trainer = classifier_trainer()
    OUTPUT_DIR = '/home/pbromley/synth-seqs-figures/'

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
                                              plot_dir=OUTPUT_DIR)

    hyper_param_search.search()
    hyper_param_search.save(OUTPUT_DIR, 'search_results.csv')

def train_model():
    trainer = classifier_trainer()

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

    FIGURE_DIR = '/home/pbromley/synth-seqs-figures/classifier-len-200/'
    MODEL_DIR = '/home/pbromley/synth-seqs-models/classifier-len-200/'

    trainer.plot(FIGURE_DIR)
    trainer.save('classifier.pth', MODEL_DIR)

if __name__ == "__main__":
    #model = conv_net_one_layer
    #model_params = {
    #    'filters': 96, 
    #    'pool_size': 100,
    #    'fully_connected': 100,
    #    'drop': 0.0,
    #}
    #model = model(**model_params)
    #from utils.net import count_parameters
    #print(f'Number of model_params for {model_params["filters"]}: {count_parameters(model)}')
    train_model() 
