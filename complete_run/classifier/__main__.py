from torch import cuda, device

from .analysis import Evaluator
from .models import conv_net
from .trainer import ClassifierTrainer, HyperParameterSearch, ParameterGroup

def main():
    dev = device("cuda" if cuda.is_available() else "cpu")

    #TODO: This is all temporary
    EPOCHS = 40 
    BATCH_SIZE = 256
    MODEL = conv_net
    DEVICE = dev
    DATA_DIR = '/home/pbromley/synth-seqs-data/'
    trainer = ClassifierTrainer(EPOCHS,
                                BATCH_SIZE,
                                MODEL,
                                DATA_DIR,
                                DEVICE)

    # OPTIMIZER_PARAMS = {
    #     'lr': 0.0018,
    #     'betas': (0.9, 0.99)
    # }
    # MODEL_PARAMS = {
    #     'filters': 64,
    #     'pool_size': 5,
    #     'fully_connected': 25,
    #     'drop': 0.0,
    # }
    # trainer.train(OPTIMIZER_PARAMS, MODEL_PARAMS)

    ### ALL ###
    optimizer_params_group = ParameterGroup({
        'lr': [0.0018],
        'betas': [(0.9, 0.99)],
    })
    ### MODEL PARAMS ###
    model_params_group = ParameterGroup({
        'filters': [32, 64],
        'pool_size': [5, 10],
        'fully_connected': [50],
        'drop': [0.0],
    })

    hyper_param_search = HyperParameterSearch(trainer,
                                              optimizer_params_group,
                                              model_params_group)

    results = hyper_param_search.search()

    print(results)


    OUTPUT_DIR = '/home/pbromley/synth-seqs-figures/'
    # evaluator = Evaluator(trainer.collector, OUTPUT_DIR)

    # evaluator.plot_loss('loss.png')
    # evaluator.plot_confusion('confusion.png')


if __name__ == "__main__":
    main()
