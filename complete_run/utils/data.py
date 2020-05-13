from os import path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.constants import (
    COMPONENTS,
    data_filename,
    SEQUENCES,
    SEQUENCE_LENGTH,
    TEST,
    TRAIN,
    VALIDATION,
)


class DHSSequencesDataset(Dataset):

    def __init__(self, seqs, components):
        self.seqs = seqs
        self.components = components

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        one_hot = self.seqs[idx]
        component = self.components[idx]
        return one_hot, component

    @property
    def component_distribution(self):
        """Use for weighted random sampling of inbalanced classes
        """
        return np.bincount(self.components.astype(int))


class DHSDataLoader:
    def __init__(self, batch_size, model, data_dir):
        self.model = model
        self.data_dir = data_dir

        self.dataloaders = self.make_dataloaders(batch_size)

    @property
    def train(self):
        return self.dataloaders[TRAIN]
    
    @property
    def test(self):
        return self.dataloaders[TEST]

    @property
    def validation(self):
        return self.dataloaders[VALIDATION]

    def load(self, label, kind):
        f = self.data_dir + data_filename(label, kind, self.model)
        assert path.exists(f), (
            f"Trying to load numpy file {f} but file does not exist. "
            f"Have you run the `make_data` module? If so, is {self.data_dir} "
            f"the directory where your numpy datasets are stored?"
        )
        return np.load(f)

    def make_dataset(self, label):
        xs = self.load(label, SEQUENCES).reshape(-1, 1, SEQUENCE_LENGTH, 4)
        ys = self.load(label, COMPONENTS)
        return DHSSequencesDataset(xs, ys)
    
    def make_dataloaders(self, batch_size):
        dataloaders = {}
        for label in [TRAIN, TEST, VALIDATION]:
            dataset = self.make_dataset(label)
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=True)
            dataloaders[label] = dataloader
        
        return dataloaders











