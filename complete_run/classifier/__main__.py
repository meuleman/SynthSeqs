from torch import cuda, device

from .analysis import Collector, Evaluator
from .models import conv_net
from .trainer import ClassifierTrainer

def main():
    dev = device("cuda" if cuda.is_available() else "cpu")

    #TODO: This is all temporary
    EPOCHS = 40 
    BATCH_SIZE = 256
    OPTIMIZER_PARAMS = {
        'lr': 0.0018,
        'betas': (0.9, 0.99)
    }
    MODEL_PARAMS = (64, 5, 25, 0.0)
    MODEL = conv_net
    COLLECTOR = Collector(dev)
    DATA_DIR = '/home/pbromley/synth-seqs-data/'
    DEVICE = dev
    trainer = ClassifierTrainer(EPOCHS,
                                BATCH_SIZE,
                                OPTIMIZER_PARAMS,
                                MODEL_PARAMS,
                                MODEL,
                                COLLECTOR,
                                DATA_DIR,
                                DEVICE)

    trainer.train()

    OUTPUT_DIR = '/home/pbromley/synth-seqs-figures/'
    evaluator = Evaluator(trainer.collector, OUTPUT_DIR)

    evaluator.plot_loss('loss.png')
    evaluator.plot_confusion('confusion.png')


if __name__ == "__main__":
    main()
