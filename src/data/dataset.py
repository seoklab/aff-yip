import sys, copy, os 
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from .graph_utils import lig_graph_gen, rec_graph_gen

# Receptor, Ligand, Affinity(absolute) value dataset
class RLADataset(Dataset):
    def __init__(self, data, config,
                split='train',
                gen_conformer=False): 
        """    
        Parameters:
        -----------
        data: dict
            A dictionary containing the following keys:
            'ligand_mol2': file path 
            'receptor_pdb': file path
            'pdbid': pdbid 
            'affinity': nM converted, only Ki Kd values
        config: from configuration - train_params 
        split: test, valid, train
        gen_conformer: generating conformer on the fly  
        """
        

    def __len__(self):
        # Return the number of samples in the dataset
        
        return len(self.pdbid)

    def __getitem__(self, index):

        ligand_file = self.ligand_data[index]

        # generate ligand graph
        LG = lig_graph_gen(ligand_file)

        # generate receptor graph 
        # NEED POCKET CENTER INFO 이 부분 추가하기 
        receptor_file = self.receptor_data[index]
        RG = rec_graph_gen(receptor_file)

        pdbid = self.pdbid[index] 
        
        # Retrieve the affinity value for this sample
        affinity = self.affinity_data[index]
        affinity = torch.tensor(self.affinity_data[index], dtype=torch.float32)


        return sample