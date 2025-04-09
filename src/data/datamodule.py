
import pandas as pd
import re
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import AbsoluteDataset
import numpy as np 
from argparse import Namespace
import pandas as pd
import re
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from .dataset import RLADataset
from .utils import skip_none_collate

class AbsoluteDataModule(pl.LightningDataModule): # for absolute affinity 
    """Initialize DataModule / absolute affinity
    
    Parameters
    ----------
    cfg: confinguration 
    """
    def __init__(self, train_params): 
        super().__init__()
        self.cfg = train_params
        self.batch_size = train_params.batch_size

    def setup(self, stage=None):

        train_data, val_data = self._load_from_csv()

        self.train_dataset = RLADataset(train_data, self.cfg, split="train") 
        self.val_dataset = RLADataset(val_data, self.cfg, split="val") 
    
    def _load_from_csv(self):
        """
        Helper function to read the CSV file and return the required data.
        
        Returns:
            tuple: Ligand data, receptor data, and affinity labels for train and validation sets.
        """
        dbtype = self.cfg.dbtype
        if dbtype == 'lppdb':
            lppdb_csv = '/home.galaxy4/j2ho/DB/LP-PDBBind/dataset/LP_PDBBind.csv'
            pdbbind_dir = '/home/j2ho/DB/pdbbind/'
            csv_path = lppdb_csv
        else: 
            raise ValueError("Only 'lppdb' is supported for now.")
        
        data = pd.read_csv(csv_path)
        
        # Extract necessary columns
        receptor_data = data['pdbid'].values
        affinity_data = self._parse_affinity_data(data['kd/ki'].values)
        affinity_data = np.array(affinity_data)
        split_data = data['new_split'].values  # either train, valid, test, or others.
        category = data['category'].values  # general, core, refined
        
        # Split data based on the 'new_split' column
        train_indices = split_data == 'train'
        val_indices = split_data == 'val'
        
        # Train and validation data
        train_data = {
            'ligand_mol2': self._get_ligand_file_paths(receptor_data[train_indices], category[train_indices], pdbbind_dir),
            'receptor_pdb': self._get_receptor_file_paths(receptor_data[train_indices], category[train_indices], pdbbind_dir),
            'pdbid': receptor_data[train_indices],
            'affinity': affinity_data[train_indices]
        }
        
        val_data = {
            'ligand_mol2': self._get_ligand_file_paths(receptor_data[val_indices], category[val_indices], pdbbind_dir),
            'receptor_pdb': self._get_receptor_file_paths(receptor_data[val_indices], category[val_indices], pdbbind_dir),
            'pdbid': receptor_data[val_indices],
            'affinity': affinity_data[val_indices],
        }
        
        return train_data, val_data
    
    def _get_ligand_file_paths(self, receptor_ids, categories, pdbbind_dir):
        ligand_paths = []
        for pdbid, cat in zip(receptor_ids, categories):
            if cat == 'general':
                ligand_file = Path(f"{pdbbind_dir}/v2020-others/{pdbid}/{pdbid}_ligand.mol2").resolve()
            else:
                ligand_file = Path(f"{pdbbind_dir}/v2020-refined/{pdbid}/{pdbid}_ligand.mol2").resolve()
            ligand_paths.append(ligand_file)
        return ligand_paths
    
    def _get_receptor_file_paths(self, receptor_ids, categories, pdbbind_dir):
        receptor_paths = []
        for pdbid, cat in zip(receptor_ids, categories):
            if cat == 'general':
                ligand_file = Path(f"{pdbbind_dir}/v2020-others/{pdbid}/{pdbid}_protein.pdb").resolve()
            else:
                ligand_file = Path(f"{pdbbind_dir}/v2020-refined/{pdbid}/{pdbid}_protein.pdb").resolve()
            receptor_paths.append(ligand_file)
        return receptor_paths
    
    def _parse_affinity_data(self, affinity_data): 
        """
        Helper function to parse the affinity value strings (e.g., "Kd=0.006uM") and convert them to nM.
        
        Args:
            affinity_data (array): Array of affinity value strings.
        
        Returns:
            list: List of parsed affinity numeric values in nM.
        """
        affinity_in_nM = []
        
        for value in affinity_data:
            match = re.search(r'(Kd|Ki)=(\d*\.?\d+)(uM|nM)', value)
            
            if match:
                numeric_value = float(match.group(2))  # Corrected to group(2) for numeric part
                unit = match.group(3).lower()  # Corrected to group(3) for unit
                
                if unit == 'um':
                    numeric_value *= 1000
                elif unit == 'nm':
                    pass
                else:
                    numeric_value = None
                
                affinity_in_nM.append(numeric_value)
            else:
                affinity_in_nM.append(None)  # Handle missing or malformed values
        
        return affinity_in_nM

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.cfg.num_workers, collate_fn=skip_none_collate)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.cfg.num_workers, collate_fn=skip_none_collate)
