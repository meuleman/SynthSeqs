import sys
import time
import numpy as np
import torch
from torch import device, cuda
from torch.optim import Adam

from generator.constants import NZ

from .optimize import SequenceTuner
from .vectors import TuningVectors


# THIS IS ALL TEMPORARY
from generator.models import snp_generator
from classifier.models import conv_net, conv_net_one_layer

def optimize(vector_id_range, target_class, verbose=False):
    dev = device("cuda" if cuda.is_available() else "cpu")

    generator_params = {
        'nz': NZ,
        'num_filters': 640,
        'len_filters': 15,
        'transpose_size': 10,
    }
    generator = snp_generator(**generator_params).to(dev)
<<<<<<< HEAD
    GENERATOR_PATH = '/home/pbromley/synth-seqs-models/generator-len-200-640filters/generator.pth'
=======
    GENERATOR_PATH = '/home/pbromley/synth-seqs-models/generator-len-200/generator.pth'
>>>>>>> 5577b00d865ab7dce789e138a89d4feb016acb8f
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=dev))
    generator.train(False)

    model_params = {           
        'filters': 100,
        'pool_size': 200,
        'fully_connected': 100,
        'drop': 0.5,
    }
 
    classifier = conv_net_one_layer(**model_params).to(dev)
    MODEL_PATH = '/home/pbromley/synth-seqs-models/classifier-len-200/classifier.pth'
    classifier.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    classifier.train(False)

    classifier.eval()
    generator.eval()

    optimizer_params = {
        'lr': 0.017,
        'betas': (0.8, 0.59)
    }

    tuner = SequenceTuner(generator,
                          classifier,
                          optimizer_params,
                          dev)

    vector_path = '/home/pbromley/projects/synth_seqs/tuning/initial/vectors-len-100.npy'
    vectors = TuningVectors()
<<<<<<< HEAD
    opt_zs = vectors.load_fixed(vector_path, slice(*vector_id_range))
    iters = 10000
    save_dir = f'/home/pbromley/projects/synth_seqs/tuning/640filters/{target_class}/'
    seed_dir = save_dir + 'seed/'

    start = time.time()
    tuner.tune(opt_zs,
               target_class,
               iters,
               save_dir,
               verbose=verbose,
               seed_dir=seed_dir,
               vector_id_range=vector_id_range)
                   
    elapsed = time.time() - start

=======
    opt_zs = vectors.load_fixed(vector_path, slice(0, vector_id))
    iters = 10000
    save_dir = f'/home/pbromley/projects/synth_seqs/tuning/global_penalty_len200/{target_class}/'

    start = time.time()
    tuner.optimize_multiple(opt_zs,
                            target_class,
                            iters,
                            save_dir) 
                   
    elapsed = time.time() - start

    #tuner.save_training_history(save_dir, vector_id)

>>>>>>> 5577b00d865ab7dce789e138a89d4feb016acb8f
def initialize_fixed_vectors(num_vectors, len_vectors, path, seed=None):
    tv = TuningVectors()
    tv.save_fixed(num_vectors, len_vectors, path, seed=seed)

if __name__ == '__main__':
<<<<<<< HEAD
    assert len(sys.argv) == 2, 'Wrong number of arguments given (exactly 1 required)'
    args = int(sys.argv[1])
    #c = args // 10
    #range_base = (args % 10) * 10000
    #vector_id_range = (range_base, range_base + 10000)

    c = args
    vector_id_range = (0, 5000)
    optimize(vector_id_range, c, verbose=True)
=======
    #assert len(sys.argv) == 2, 'Wrong number of arguments given (exactly 1 required)'

    c = int(sys.argv[1])
    #vector_id = args % 1000
    #target_component = args // 1000

    optimize(1000, c)
>>>>>>> 5577b00d865ab7dce789e138a89d4feb016acb8f

