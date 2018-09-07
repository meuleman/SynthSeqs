import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision.datasets as dset



# Setting up the data (custom collate function for variable length inputs w/in batch)
#   https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/2

# Custom dataset class to handle loading in of list of numpy arrays
#   Makes it so don't have to write arrays to files and use DatasetFolder
class DHSSequencesDataset(Dataset):
    """DHS sequences of varying length, as well as a component label"""

    def __init__(self, seqs, components, transform=None):
        """
        Args:
            seqs (list/np.array): List of one-hot numpy array DNA sequences
            components (list/np.array): List of integers indicating components 1-15
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.seqs = seqs
        self.components = components
        self.transform = transform

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        one_hot = self.seqs[idx]
        component = self.components[idx]

        if self.transform:
            image = self.transform(one_hot)

        return one_hot, component


# For loading batches with variable length inputs
def collate_variable_length(batch):
    one_hot = [sample[0] for sample in batch]
    component = [sample[1] for sample in batch]
    component = torch.LongTensor(component)
    return [one_hot, component]


def load_the_data(filter_type="mean_signal", rc=False):
    # Read in data
    if filter_type == "mean_signal":
        x_train = np.load("/home/pbromley/generative_dhs/data_numpy/ms_one_hot_train.npy")
        x_test = np.load("/home/pbromley/generative_dhs/data_numpy/ms_one_hot_test.npy")
        if rc:
            x_train_rc = np.load("/home/pbromley/generative_dhs/data_numpy/ms_one_hot_train_rc.npy")
            x_test_rc = np.load("/home/pbromley/generative_dhs/data_numpy/ms_one_hot_test_rc.npy")
            x_train = np.concatenate([x_train, x_train_rc])
            x_test = np.concatenate([x_test, x_test_rc])
        y_train = np.zeros(len(x_train))
        y_test = np.zeros(len(x_test)) 
    else:
        x_train = np.load("/home/pbromley/generative_dhs/data_numpy/strong_one_hot_train.npy")
        y_train = np.load("/home/pbromley/generative_dhs/data_numpy/strong_components_train.npy")
        x_test = np.load("/home/pbromley/generative_dhs/data_numpy/strong_one_hot_test.npy")
        y_test = np.load("/home/pbromley/generative_dhs/data_numpy/strong_components_test.npy")
        if rc:
            x_train_rc = np.load("/home/pbromley/generative_dhs/data_numpy/strong_one_hot_train_rc.npy")
            x_test_rc = np.load("/home/pbromley/generative_dhs/data_numpy/strong_one_hot_test_rc.npy")
            x_train = np.concatenate([x_train, x_train_rc])
            x_test = np.concatenate([x_test, x_test_rc])
            y_train = np.concatenate([y_train, y_train])
            y_test = np.concatenate([y_test, y_test])

    if (filter_type == "z_score"):
        y_train = y_train - 1       # raw component data goes from 1-15, want 0-14
        y_test = y_test - 1         # "       "       "    "    "    "     "    "

    return x_train, y_train, x_test, y_test


def load_nmf_and_sig_data():
    nmf_train = np.load("/home/pbromley/generative_dhs/data_numpy/strong_nmfs_train.npy")
    nmf_test = np.load("/home/pbromley/generative_dhs/data_numpy/strong_nmfs_test.npy")
    sig_train = np.load("/home/pbromley/generative_dhs/data_numpy/strong_sig_train.npy")
    sig_test = np.load("/home/pbromley/generative_dhs/data_numpy/strong_sig_test.npy")
    return nmf_train, nmf_test, sig_train, sig_test


def get_the_dataloaders(batch_size, binary_class=None, weighted_sample=False, one_dim=False, data_type='mean_signal', rc=False):

    x_train, y_train, x_test, y_test = load_the_data(filter_type=data_type, rc=rc)
    if one_dim:
        x_train = x_train.transpose(0, 2, 1)
        x_test = x_test.transpose(0, 2, 1)
    else:
        x_train = x_train.reshape(-1, 1, 100, 4)
        x_test = x_test.reshape(-1, 1, 100, 4)
    if binary_class is not None:
        y_train = (y_train == binary_class).astype(int)
        y_test = (y_test == binary_class).astype(int)
    class_dist_train = np.bincount(y_train.astype(int))
    class_dist_test = np.bincount(y_test.astype(int))

    dhs_dataset_train = DHSSequencesDataset(x_train, y_train)
    dhs_dataset_test = DHSSequencesDataset(x_test, y_test)

    ### Sample balanced class distribution
    class_weights = np.sum(class_dist_train) / class_dist_train    # calculate inverse probs
    weights = [class_weights[int(comp)] for _, comp in dhs_dataset_train]    # assign weight for every sample
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))  # create sampler

    class_weights = np.sum(class_dist_test) / class_dist_test
    weights = [class_weights[int(comp)] for _, comp in dhs_dataset_test]
    sampler_test = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    ###### For Uniform Length ######
    if weighted_sample:
        dhs_dataloader_train = DataLoader(dataset=dhs_dataset_train,
                                batch_size=batch_size,
                                sampler=sampler_train)
        dhs_dataloader_test = DataLoader(dataset=dhs_dataset_test,
                                batch_size=batch_size,
                                shuffle=True)
    else:
        dhs_dataloader_train = DataLoader(dataset=dhs_dataset_train,
                                batch_size=batch_size,
                                shuffle=True)
        dhs_dataloader_test = DataLoader(dataset=dhs_dataset_test,
                                batch_size=batch_size,
                                shuffle=True)


    ###### For Variable Length ######
    # dhs_dataloader = DataLoader(dataset=dhs_dataset,
    #                             batch_size=4,
    #                             shuffle=True,
    #                             collate_fn=collate_variable_length)
    return dhs_dataloader_train, dhs_dataloader_test
