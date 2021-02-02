import torch
from torch.utils.data import Dataset, DataLoader, sampler
import pandas as pd
from functools import lru_cache
import numpy as np


class DataSplit:

    def __init__(self, dataset, test_train_split=0.8, val_train_split=0.1, shuffle=False):
        self.dataset = dataset

        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))

        if shuffle:
            np.random.shuffle(self.indices)

        train_indices, self.test_indices = self.indices[:test_split], self.indices[test_split:]
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))

        self.train_indices, self.val_indices = train_indices[: validation_split], train_indices[validation_split:]

        self.train_sampler = sampler.SubsetRandomSampler(self.train_indices)
        self.val_sampler = sampler.SubsetRandomSampler(self.val_indices)
        self.test_sampler = sampler.SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=50, num_workers=4):
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=50, num_workers=4):
        self.train_loader = DataLoader(self.dataset, batch_size=batch_size,
                                       sampler=self.train_sampler, shuffle=False, num_workers=num_workers)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=50, num_workers=4):
        self.val_loader = DataLoader(self.dataset, batch_size=batch_size,
                                     sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=50, num_workers=4):
        self.test_loader = DataLoader(self.dataset, batch_size=batch_size,
                                      sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
        return self.test_loader


class ETDData(Dataset):
    def __init__(self, data, feature_list, objective):
        """
        Args:
            csv_path (string): path to csv file
            feature_list (string): List of features to consider
        """
        # Read the csv file
        self.data = data
        # Set up samples
        if feature_list is not None:
            self.samples = np.asarray(self.data[feature_list])
        else:
            self.samples = np.asarray(self.data.loc[:, self.data.columns != 'atd'])
	    
        #Normalize 
        self.normalized_samples = (self.samples - np.mean(self.samples, axis=0)) / np.std(self.samples, axis=0)
        self.normalized_samples = np.nan_to_num(self.normalized_samples)
        # Set up labels
        if objective == "ae":
            self.labels = np.asarray(self.normalized_samples)
        else:
            self.labels = np.asarray(self.data['atd'] - self.data['etd'])
 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get sample from the pandas df
        sample = self.normalized_samples[index]
        # Get label
        label = self.labels[index]
        return sample, label


    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
