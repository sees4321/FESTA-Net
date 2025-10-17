import numpy as np
import random
import torch

from torch.utils.data import DataLoader, Dataset

class CustomDataSet(Dataset):
    r"""
    A general-purpose PyTorch Dataset class that can handle an arbitrary number of tensors.

    Args:
        *tensors: A sequence of tensors that must have the same size in the first dimension.
                  The last tensor should be the label.
    """
    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("At least one tensor must be provided.")
        first_len = tensors[0].size(0)
        assert all(t.size(0) == first_len for t in tensors), "All tensors must have the same size in the first dimension."
        
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    
class DataModule_OpenAcessDatasets():
    r'''
    Create dataset for leave-one-subject-out cross-validation

    Args:
        path (str): path for the original data.
        data_mode (int): 0 - EEG & fNIRS, 1 - only EEG, 2 - only fNIRS
        label_type (int): 0 - WG, 1 - DSR, 2 - N-back
        num_val (int): number of subjects for validation. (default: 3)
        batch_size (int): batch size of the dataloader. (default: 16)
        transform_eeg (function): transform function for the EEG data (default: None)
        transform_fnirs (function): transform function for the fNIRS data (default: None)
    '''
    def __init__(self, 
                 path:str,
                 data_mode:int = 0,
                 label_type:int = 0,
                 num_val:int = 3,
                 batch_size:int = 16,
                 transform_eeg = None,
                 transform_fnirs = None,
                 ):
        super().__init__()
        assert data_mode in [0,1,2], 'data_mode should be 0 (EEG & fNIRS), 1 (EEG), or 2 (fNIRS)'
        assert label_type in [0,1,2], 'label_type should be 0 (WG), 1 (DSR), or 2 (N-back)'

        self.data_mode = data_mode
        self.label_type = label_type
        self.num_val = num_val
        self.batch_size = batch_size
        self.test_idx = 0
        
        # load data
        if label_type == 0:
            name = 'WG'
        elif label_type == 1:
            name = 'dsr'
        elif label_type == 2:
            name = 'nback'
        data = np.load(f'{path}/{name}.npz')
        self.eeg = data['eeg'] # (subjects, trials, channels, time samples)
        self.fnirs = data['fnirs'] # (subjects, trials, channels, time samples)
        self.label = data['label'] # (subjects, trials)
        self.subjects = [i for i in range(self.eeg.shape[0])]

        # data shape
        self.data_shape_eeg = list(self.eeg.shape[-2:])
        self.data_shape_fnirs = list(self.fnirs.shape[-2:])

        # if there is extra preprocessing for EEG and fNIRS data
        if transform_eeg:
            self.eeg = transform_eeg(self.eeg)
        if transform_fnirs:
            self.fnirs = transform_fnirs(self.fnirs)
    
    def __len__(self):
        return self.subjects

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.test_idx < len(self.subjects):
            eeg_torch = torch.from_numpy(self.eeg[self.subjects[self.test_idx]]).float()
            fnirs_torch = torch.from_numpy(self.fnirs[self.subjects[self.test_idx]]).float()
            label_torch = torch.from_numpy(self.label[self.subjects[self.test_idx]]).long()
            test_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch, batch_size=64)

            train_subjects, val_subjects = self.train_val_split()
            eeg_torch = torch.from_numpy(np.concatenate([self.eeg[i] for i in train_subjects])).float()
            fnirs_torch = torch.from_numpy(np.concatenate([self.fnirs[i] for i in train_subjects])).float()
            label_torch = torch.from_numpy(np.concatenate([self.label[i] for i in train_subjects])).long()
            train_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch)

            self.test_idx += 1
            if len(val_subjects) > 0:
                eeg_torch = torch.from_numpy(np.concatenate([self.eeg[i] for i in val_subjects])).float()
                fnirs_torch = torch.from_numpy(np.concatenate([self.fnirs[i] for i in val_subjects])).float()
                label_torch = torch.from_numpy(np.concatenate([self.label[i] for i in val_subjects])).long()
                val_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch, True)

                return train_loader, val_loader, test_loader
            else:
                return train_loader, None, test_loader
        else:
            raise StopIteration
    
    def create_dataloader(self, eeg, fnirs, label, shuffle=False, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        if self.data_mode == 0:
            return DataLoader(CustomDataSet(eeg, fnirs, label), batch_size, shuffle=shuffle)
        elif self.data_mode == 1:
            return DataLoader(CustomDataSet(eeg, label), batch_size, shuffle=shuffle)
        elif self.data_mode == 2:
            return DataLoader(CustomDataSet(fnirs, label), batch_size, shuffle=shuffle)
        
    def train_val_split(self):
        subj = [i for i in self.subjects if i != self.subjects[self.test_idx]]
        random.shuffle(subj)
        return subj[self.num_val:], subj[:self.num_val]