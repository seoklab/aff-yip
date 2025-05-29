# cluade aided outline for dataset 
# NOT FIT FOR ACTUAL USAGE YET !!!!

import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class ProteinDataset(Dataset):
    """
    Dataset for protein structure data
    """
    def __init__(self, data_path, mode='train', max_length=500):
        """
        Initialize the dataset
        :param data_path: Path to the data directory
        :param mode: One of 'train', 'val', or 'test'
        :param max_length: Maximum sequence length
        """
        self.data_path = data_path
        self.mode = mode
        self.max_length = max_length
        
        # Load the dataset
        file_path = os.path.join(data_path, f"{mode}.npz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        
        data = np.load(file_path, allow_pickle=True)
        
        # Store the data
        self.X = data['X']  # Backbone coordinates [N, L, 4, 3]
        self.S = data['S']  # Sequences [N, L]
        
        # For MQA data, also load quality scores if present
        if 'y' in data:
            self.y = data['y']  # Quality scores [N]
            self.has_quality = True
        else:
            self.has_quality = False
            
        # Filter by length
        self.valid_indices = [i for i, x in enumerate(self.X) 
                             if x.shape[0] <= max_length]
        
        print(f"Loaded {len(self.valid_indices)} proteins for {mode}")
        
    def __len__(self):
        """Return the number of proteins in the dataset"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Get a protein sample"""
        idx = self.valid_indices[idx]
        
        # Get the protein data
        X = self.X[idx]  # [L, 4, 3]
        S = self.S[idx]  # [L]
        
        # Create a mask (1 for positions with data, 0 for padding)
        L = X.shape[0]
        mask = np.ones(L, dtype=np.bool_)
        
        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32)
        S = torch.tensor(S, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.bool)
        
        if self.has_quality:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return X, S, mask, y
        else:
            return X, S, mask


class ProteinCollator:
    """
    Collates protein samples into a batch
    """
    def __init__(self, has_quality=False):
        self.has_quality = has_quality
    
    def __call__(self, batch):
        """
        Collate protein samples
        :param batch: List of protein samples
        :return: Batched tensors
        """
        if self.has_quality:
            X, S, mask, y = zip(*batch)
        else:
            X, S, mask = zip(*batch)
        
        # Get sequence lengths
        lengths = [x.shape[0] for x in X]
        max_length = max(lengths)
        
        # Pad the sequences
        X_padded = []
        S_padded = []
        mask_padded = []
        
        for i, l in enumerate(lengths):
            # Padding for X
            padding = torch.zeros(max_length - l, 4, 3)
            X_padded.append(torch.cat([X[i], padding], dim=0))
            
            # Padding for S
            padding = torch.zeros(max_length - l, dtype=torch.long)
            S_padded.append(torch.cat([S[i], padding], dim=0))
            
            # Padding for mask
            padding = torch.zeros(max_length - l, dtype=torch.bool)
            mask_padded.append(torch.cat([mask[i], padding], dim=0))
        
        # Stack
        X_batch = torch.stack(X_padded, dim=0)
        S_batch = torch.stack(S_padded, dim=0)
        mask_batch = torch.stack(mask_padded, dim=0)
        
        if self.has_quality:
            y_batch = torch.tensor(y, dtype=torch.float32)
            return X_batch, S_batch, mask_batch, y_batch
        else:
            return X_batch, S_batch, mask_batch


class ProteinDataModule(pl.LightningDataModule):
    """
    Data module for protein structure data
    """
    def __init__(self, data_path, batch_size=8, num_workers=8, max_length=500,
                 is_mqa=True):
        """
        Initialize the data module
        :param data_path: Path to the data directory
        :param batch_size: Batch size
        :param num_workers: Number of workers for data loading
        :param max_length: Maximum sequence length
        :param is_mqa: Whether the data is for Model Quality Assessment
        """
        super(ProteinDataModule, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.is_mqa = is_mqa
        
    def setup(self, stage=None):
        """
        Set up the datasets
        :param stage: Stage ('fit', 'validate', 'test', or None)
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = ProteinDataset(
                self.data_path, mode='train', max_length=self.max_length
            )
            self.val_dataset = ProteinDataset(
                self.data_path, mode='val', max_length=self.max_length
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = ProteinDataset(
                self.data_path, mode='test', max_length=self.max_length
            )
    
    def train_dataloader(self):
        """Get the training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ProteinCollator(has_quality=self.is_mqa),
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Get the validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ProteinCollator(has_quality=self.is_mqa),
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Get the test data loader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ProteinCollator(has_quality=self.is_mqa),
            pin_memory=True
        )