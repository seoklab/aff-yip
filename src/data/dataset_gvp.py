# src/data/dataset_gvp.py
import os
import torch
# import torch_geometric # No longer directly used for Data creation here
# import torch_cluster # No longer directly used here
import numpy as np
from torch.utils.data import Dataset

# Import your structures, and new featurizer classes
from .structures import Protein, Ligand, VirtualNode # Assuming this path is correct for structures
from .featurizers.protein import ProteinFeaturizer
from .featurizers.ligand import LigandFeaturizer


class RLADataset(Dataset):
    def __init__(self, data_path=None, target_dict: dict = None, mode='train', top_k=30, crop_size=30): # max_length not used in snippet
        self.data_path = data_path
        self.mode = mode
        self.top_k = top_k 
        self.crop_size = crop_size 
        self.samples = []
        self.skip_virtual = False
        # Instantiate featurizers
        self.protein_featurizer = ProteinFeaturizer(top_k=self.top_k)
        self.ligand_featurizer = LigandFeaturizer()

        # Default protein and ligand for testing mode (if target_dict is None)
        # These are loaded once and used if no target_dict is provided.
        self.default_protein_w_obj = Protein(
            pdb_filepath=os.path.join(data_path, '2etr.pdb'),
            read_water=True, read_ligand=False, read_chain=['A']
        )
        self.default_protein_obj = Protein(
            pdb_filepath=os.path.join(data_path, '2etr.pdb'),
            read_water=False, read_ligand=False, read_chain=['A'] # No water, no ligand, only chain A
        )
        self.default_ligand_obj = Ligand(
            mol2_filepath=os.path.join(data_path, '2etr_lig.mol2') # Consider making this path relative or configurable
        )

        if target_dict:
            for target_info in target_dict: # Renamed for clarity
                try:
                    pdbfile = target_info.get('pdb_file_biolip', None) # biolip pdbs, only residues
                    pdbfile_raw = target_info.get('pdb_file_db', None) # pdb files from db, with water and other residues
                    receptor_chain = target_info.get('receptor_chain', None) # Optional receptor chain
                    ligfile = target_info.get('ligand_mol2', None)
                    affinity_value = target_info.get('affinity', None)
                    if affinity_value is not None:
                        affinity = torch.tensor(float(affinity_value), dtype=torch.float32)
                    name = target_info.get('name', 'UnnamedTarget')
                    center = target_info.get('center', None) # Optional center for cropping
                    center = torch.tensor(center, dtype=torch.float32) if center is not None else None

                    if not pdbfile or not ligfile:
                        print(f"[Warning] Skipped {name}: Missing PDB or ligand file path.")
                        continue

                    protein_w_obj = Protein(pdb_filepath=pdbfile_raw, read_water=True, read_ligand=False, read_chain=[receptor_chain] if receptor_chain else [])
                    protein_obj = Protein(pdb_filepath=pdbfile, read_water=False, read_ligand=False)
                    ligand_obj = Ligand(mol2_filepath=ligfile, drop_H=False)
                    # protein graph with virtual nodes and water
                    # print ('Featurizing protein with virtual nodes and water...')
                    if not self.skip_virtual:
                        protein_virtual_graph = self.protein_featurizer.featurize_graph_with_virtual_nodes(
                        protein_w_water=protein_w_obj,
                        protein_wo_water=protein_obj, # Virtual nodes based on default protein
                        ligand=ligand_obj,   # and default ligand
                        center=center,
                        crop_size=self.crop_size,
                        target_name=name
                        )
                    else:
                        protein_virtual_graph = None
                    """
                    featurize_graph_with_water handles has_water check 
                    if protein_obj has water, it will include it in the graph
                    """
                    # These three are equivalent / protein graph without water and without virtual nodes
                    # print ('Featurizing protein only, using no_water_protein object') 
                    # protein_only_graph = self.protein_featurizer.featurize_graph_with_water(protein_obj, center=center, crop_size=self.crop_size)
                    # print ('Featurizing protein only, using no_water_protein object, with no_water function')
                    protein_only_graph = self.protein_featurizer.featurize_no_water_graph(protein_obj, center=center, crop_size=self.crop_size)  
                    # print ('Featurizing protein only, using water_protein object with no_water function')
                    # protein_only_graph = self.protein_featurizer.featurize_no_water_graph(protein_w_obj, center=center, crop_size=self.crop_size)    

                    ## protein graph with water but without virtual nodes
                    # print ('Featurizing protein with water (no virtual nodes), using water_protein object')
                    protein_water_graph = self.protein_featurizer.featurize_graph_with_water(protein_w_obj, center=center, crop_size=self.crop_size)
                    
                    ligand_graph = self.ligand_featurizer.featurize_graph(ligand_obj)
                    # ligand_graph.name = name 
                    # if any of the graphs are None, skip this sample
                    if not self.skip_virtual:
                        if protein_virtual_graph is None or protein_water_graph is None or protein_only_graph is None or ligand_graph is None:
                            print(f"[Warning] Skipped {name}: One or more graphs could not be created.")
                            continue
                        self.samples.append({
                            'name': name,
                            'protein_virtual': protein_virtual_graph, # Protein graph + water + virtual nodes
                            'protein_water': protein_water_graph, # Protein gra
                            'protein_only': protein_only_graph, # Protein - water - virtual nodes
                            'ligand': ligand_graph,
                            'affinity': affinity,
                        })
                    else:
                        if protein_water_graph is None or protein_only_graph is None or ligand_graph is None:
                            print(f"[Warning] Skipped {name}: One or more graphs could not be created.")
                            continue
                        self.samples.append({
                            'name': name,
                            'protein_water': protein_water_graph, # Protein graph + water
                            'protein_only': protein_only_graph, # Protein - water
                            'ligand': ligand_graph,
                            'affinity': affinity,
                        })
                except Exception as e:
                    target_name_for_error = target_info.get('name', 'Unknown Target')
                    print(f"[Warning] Skipped {target_name_for_error} due to error: {e}")
        else:
            # Testing mode with default protein and ligand
            center = torch.tensor([50, 100, 30], dtype=torch.float32)
            print ('Featurizing protein with virtual nodes and water...')
            protein_virtual_graph = self.protein_featurizer.featurize_graph_with_virtual_nodes(
                protein_w_water=self.default_protein_w_obj,
                protein_wo_water=self.default_protein_obj, # Virtual nodes based on default protein
                ligand=self.default_ligand_obj,   # and default ligand
                center=center,
                crop_size=self.crop_size,
                target_name='default_target'  # Default target name for testing
            )
            """
            featurize_graph_with_water handles has_water check 
            if protein_obj has water, it will include it in the graph
            """
            # These three are equivalent / protein graph without water and without virtual nodes
            print ('Featurizing protein only, using no_water_protein object') 
            protein_only_graph = self.protein_featurizer.featurize_graph_with_water(self.default_protein_obj, center=center, crop_size=self.crop_size)  
            print ('Featurizing protein only, using no_water_protein object, with no_water function')
            protein_only_graph = self.protein_featurizer.featurize_no_water_graph(self.default_protein_obj, center=center, crop_size=self.crop_size) 
            print ('Featurizing protein only, using water_protein object with no_water function')
            protein_only_graph = self.protein_featurizer.featurize_no_water_graph(self.default_protein_w_obj, center=center, crop_size=self.crop_size) 

            ## protein graph with water but without virtual nodes
            print ('Featurizing protein with water (no virtual nodes), using water_protein object')
            protein_water_graph = self.protein_featurizer.featurize_graph_with_water(self.default_protein_w_obj, center=center, crop_size=self.crop_size)
            
            ligand_graph = self.ligand_featurizer.featurize_graph(self.default_ligand_obj)
            
            data_sample = { 
                'protein_virtual': protein_virtual_graph, # Protein graph + water + virtual nodes
                'protein_water': protein_water_graph, # Protein gra
                'protein_only': protein_only_graph, # Protein - water - virtual nodes
                'ligand': ligand_graph,

            }
            self.samples.append(data_sample)

    def __len__(self):
        return len(self.samples)

    # def __getitem__(self, idx):
    #     return self.samples[idx]
    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            # sample.pop('name', None)  # Remove 'name' key if it exists
            # for key, value in sample.items():
            #     print(f"[{idx}] Key: {key}, Type: {type(value)}, Dtype: {getattr(value, 'dtype', None)}")
            return sample
        except Exception as e:
            # print(f"[Dataset] Skipping sample {idx} due to error: {e}")
            return None
        
if __name__ == '__main__':
    # Example usage
    ProteinDataset = RLADataset(data_path='data', mode='train')
