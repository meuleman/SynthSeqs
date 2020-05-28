from torch import device, cuda
from torch.optim import Adam

from .optimize import SequenceTuner


# THIS IS ALL TEMPORARY
from generator.models import snp_generator_2d_temp_2a
from classifier.models import conv_net

if __name__ == '__main__':
    dev = device("cuda" if cuda.is_available() else "cpu") 

    generator = snp_generator_2d_temp_2a(100, 320, 11)
    generator.load_state_dict(torch.load())
    generator.train(False)
    generator.to(dev)


    model_params = {           
        'filters': (96, 16),
        'pool_size': 5,
        'fully_connected': 100,
        'drop': 0.55,
    }
 
    classifier = conv_net(**model_params)
    MODEL_PATH = '/home/pbromley/synth-seqs-models/classifier/model.pth'
    classifier.load_state_dict(torch.load(MODEL_PATH))
    classifier.train(False)
    classifier.to(dev)

    generator.eval()
    classifier.eval()

    optimizer = Adam
    optimizer_params = {
        'lr': 0.001,
        'betas': (0.9, 0.99)
    }

    tuner = SequenceTuner(generator,
                          classifier,
                          optimizer,
                          optimizer_params,
                          dev)

    opt_z = torch.from_numpy(np.random.normal(0, 1, 100))
    target_class = 8
    iters = 1000

    opt_z, one_hot = tuner.optimize(opt_z, target_class, iters)


