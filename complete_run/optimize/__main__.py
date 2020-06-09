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
from classifier.models import conv_net

def optimize(vector_id, target_class):
    dev = device("cuda" if cuda.is_available() else "cpu")

    generator_params = {
        'nz': NZ,
        'num_filters': 320,
        'len_filters': 15,
        'transpose_size': 10,
    }
    generator = snp_generator(**generator_params).to(dev)
    GENERATOR_PATH = '/home/pbromley/synth-seqs-models/generator/generator.pth'
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=dev))
    generator.train(False)

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

    generator.eval()
    classifier.eval()


    lrs = np.arange(0.0001, 0.05, 0.001)
    beta1s = np.arange(0.5, 1.0, 0.1)
    beta2s = np.arange(0.59, 1.0, 0.1)
    for lr in lrs:
        for beta1 in beta1s:
            for beta2 in beta2s:
                optimizer_params = {
                    'lr': lr,
                    'betas': (beta1, beta2)
                }

                tuner = SequenceTuner(generator,
                                      classifier,
                                      optimizer_params,
                                      dev)

                vector_path = '/home/pbromley/projects/synth_seqs/tuning/initial/vectors.npy'
                vectors = TuningVectors()
                opt_z = vectors.load_fixed(vector_path, vector_id) 
                iters = 4000
                save_path = f'/home/pbromley/projects/synth_seqs/tuning/optimized/{target_class}/{vector_id}.fasta'


                start = time.time()
                _, _, loss, loss_vector = tuner.optimize(opt_z, target_class, iters, save_path)
                elapsed = time.time() - start
                #print(f'ID: {vector_id}, Class: {target_class}, Iters: {iters}, Loss: {loss}, Time: {elapsed}')
                other_loss = loss_vector[np.delete(np.arange(16), target_class)]
                other_class_avg = other_loss.mean()
                other_class_min = other_loss.min()
                other_class_min_which = loss_vector.argmin()
                print(f'{lr}_{beta1}_{beta2}\t{loss}\t{other_class_avg}\t{other_class_min}\t{other_class_min_which}\t{elapsed}')



def initialize_fixed_vectors(num_vectors, len_vectors, path, seed=None):
    tv = TuningVectors()
    tv.save_fixed(num_vectors, len_vectors, path, seed=seed)

if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Wrong number of arguments given (exactly 1 required)'

    args = int(sys.argv[1])
    vector_id = args % 1000
    target_component = args // 1000


    print('params\tloss\tavg_other\tmin_other\twhich_min\ttime')
    optimize(vector_id, target_component)

