import numpy as np


class TuningVectors:

    def save_fixed(self, num_vectors, len_vectors, path, seed=None):
        if seed:
            np.random.seed(seed)
        vectors = np.random.normal(0, 1, (num_vectors, len_vectors))
        np.save(path, vectors)

    def load_fixed(self, path, which):
        return np.load(path)[which]

    def load_random(self, num_vectors, len_vectors, seed=None):
        if seed:
            np.random.seed(seed)
        return np.random.normal(0, 1, (num_vectors, len_vectors))

