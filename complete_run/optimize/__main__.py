import time
import numpy as np
import torch
from torch import device, cuda
from torch.optim import Adam
import torch.multiprocessing as mp

from .optimize import SequenceTuner


# THIS IS ALL TEMPORARY
from generator.models import snp_generator_2d_temp_2a
from classifier.models import conv_net

if __name__ == '__main__':
    dev = device("cuda" if cuda.is_available() else "cpu") 

    generator = snp_generator_2d_temp_2a(100, 320, 11).to(dev)
    GENERATOR_PATH = '/home/pbromley/synth-seqs-models/generator/generator.pth'
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=dev))
    generator.train(False)
    generator.share_memory()

    model_params = {           
        'filters': (96, 16),
        'pool_size': 5,
        'fully_connected': 100,
        'drop': 0.6,
    }
 
    classifier = conv_net(**model_params).to(dev)
    MODEL_PATH = '/home/pbromley/synth-seqs-models/classifier/classifier.pth'
    classifier.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    classifier.train(False)
    classifier.share_memory()

    generator.eval()
    classifier.eval()

    optimizer_params = {
        'lr': 0.017,
        'betas': (0.8, 0.59)
    }

    tuner = SequenceTuner(generator,
                          classifier,
                          optimizer_params,
                          dev)

    n_seqs = 100
    opt_z = torch.from_numpy(np.random.normal(0, 1, (n_seqs, 100))).float()
    target_class = 8 
    iters = 1000

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    pool = mp.Pool(processes=8)

    start = time.time()
    results = pool.starmap(tuner.optimize, [(z, target_class, iters) for z in opt_z])
    elapsed = time.time() - start
    print(f'Time: {elapsed}')

    #print(results)
