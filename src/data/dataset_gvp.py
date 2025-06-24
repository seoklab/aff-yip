# src/data/dataset_gvp.py
import os
import torch
# import torch_geometric # No longer directly used for Data creation here
# import torch_cluster # No longer directly used here
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
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
                    ligfile_for_vn = target_info.get('ligfile_for_vn', None) # Optional ligand for virtual nodes
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
                    if not self.skip_virtual:
                        protein_virtual_graph = self.protein_featurizer.featurize_graph_with_virtual_nodes(
                        protein_w_water=protein_w_obj,
                        protein_wo_water=protein_obj, # Virtual nodes based on default protein
                        ligand=ligand_obj,   # and default ligand
                        ligand_for_vn=None if ligfile_for_vn is None else Ligand(mol2_filepath=ligfile_for_vn, drop_H=False),
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
                    protein_only_graph = self.protein_featurizer.featurize_no_water_graph(protein_obj, center=center, crop_size=self.crop_size)  
                    # print ('Featurizing protein only, using water_protein object with no_water function')
                    # protein_only_graph = self.protein_featurizer.featurize_no_water_graph(protein_w_obj, center=center, crop_size=self.crop_size)    

                    ## protein graph with water but without virtual nodes
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


class RLADatasetLazy_v1(Dataset):
    def __init__(self, data_path=None, target_dict: dict = None, mode='train', top_k=30, crop_size=30, validate_samples=True):
        self.data_path = data_path
        self.mode = mode
        self.top_k = top_k 
        self.crop_size = crop_size 
        self.skip_virtual = False
        self.validate_samples = validate_samples
        
        # Store only metadata for lazy loading
        self.sample_metadata = []
        
        # Instantiate featurizers (lightweight)
        self.protein_featurizer = ProteinFeaturizer(top_k=self.top_k)
        self.ligand_featurizer = LigandFeaturizer()

        # Default paths for testing mode
        self.default_protein_w_path = os.path.join(data_path, '2etr.pdb') if data_path else None
        self.default_protein_path = os.path.join(data_path, '2etr.pdb') if data_path else None
        self.default_ligand_path = os.path.join(data_path, '2etr_lig.mol2') if data_path else None

        if target_dict and len(target_dict) > 0:
            # Real data mode - process target_dict
            self._process_target_dict(target_dict)
        else:
            # Test/default mode - use default files
            self._create_default_metadata()
            
        print(f"[Dataset] Initialized {self.__class__.__name__} with {len(self.sample_metadata)} valid samples in {mode} mode")

    def _process_target_dict(self, target_dict):
        """Process real data from target_dict"""
        for target_info in target_dict:
            try:
                # Handle both JSON and CSV formats by checking available keys
                pdbfile = target_info.get('pdb_file_biolip', target_info.get('pdb_file', None))
                pdbfile_raw = target_info.get('pdb_file_db', target_info.get('pdb_file', None))
                receptor_chain = target_info.get('receptor_chain', None)
                ligfile = target_info.get('ligand_mol2', target_info.get('ligand_file', None))
                ligfile_for_vn = target_info.get('ligfile_for_vn', None)
                affinity_value = target_info.get('affinity', None)
                name = target_info.get('name', 'UnnamedTarget')
                center = target_info.get('center', None)

                # Check required fields
                if not pdbfile or not ligfile:
                    print(f"[Warning] Skipped {name}: Missing PDB or ligand file path.")
                    continue

                # Validate file existence if requested
                if self.validate_samples:
                    missing_files = []
                    if not os.path.exists(pdbfile):
                        missing_files.append(f"PDB: {pdbfile}")
                    if pdbfile_raw and pdbfile_raw != pdbfile and not os.path.exists(pdbfile_raw):
                        missing_files.append(f"Raw PDB: {pdbfile_raw}")
                    if not os.path.exists(ligfile):
                        missing_files.append(f"Ligand: {ligfile}")
                    if ligfile_for_vn and not os.path.exists(ligfile_for_vn):
                        missing_files.append(f"VN Ligand: {ligfile_for_vn}")
                    
                    if missing_files:
                        print(f"[Warning] Skipped {name}: Missing files - {', '.join(missing_files)}")
                        continue

                # Create metadata
                metadata = {
                    'name': name,
                    'pdb_file_biolip': pdbfile,
                    'pdb_file_db': pdbfile_raw if pdbfile_raw else pdbfile,
                    'receptor_chain': receptor_chain,
                    'ligand_mol2': ligfile,
                    'ligfile_for_vn': ligfile_for_vn,
                    'affinity': torch.tensor(float(affinity_value), dtype=torch.float32) if affinity_value is not None else None,
                    'center': torch.tensor(center, dtype=torch.float32) if center is not None else None,
                    'is_default': False
                }
                self.sample_metadata.append(metadata)
                
            except Exception as e:
                target_name_for_error = target_info.get('name', 'Unknown Target')
                print(f"[Warning] Skipped {target_name_for_error} due to error: {e}")

    def _create_default_metadata(self):
        """Create default metadata for testing"""
        if not self.data_path:
            print("[Warning] No data_path provided and no target_dict - cannot create default sample")
            return
            
        # Check if default files exist
        if self.validate_samples:
            missing_defaults = []
            if not os.path.exists(self.default_protein_path):
                missing_defaults.append(f"Default protein: {self.default_protein_path}")
            if not os.path.exists(self.default_protein_w_path):
                missing_defaults.append(f"Default protein with water: {self.default_protein_w_path}")
            if not os.path.exists(self.default_ligand_path):
                missing_defaults.append(f"Default ligand: {self.default_ligand_path}")
            
            if missing_defaults:
                print(f"[Warning] Cannot create default sample: {', '.join(missing_defaults)}")
                return

        # Create default metadata
        center = torch.tensor([50, 100, 30], dtype=torch.float32)
        default_metadata = {
            'name': 'default_target',
            'pdb_file_biolip': self.default_protein_path,
            'pdb_file_db': self.default_protein_w_path,
            'receptor_chain': 'A',
            'ligand_mol2': self.default_ligand_path,
            'ligfile_for_vn': None,
            'affinity': None,
            'center': center,
            'is_default': True
        }
        self.sample_metadata.append(default_metadata)

    def __len__(self):
        return len(self.sample_metadata)

    def _create_protein_objects(self, metadata):
        """Create protein objects from metadata"""
        pdbfile = metadata['pdb_file_biolip']
        pdbfile_raw = metadata['pdb_file_db']
        receptor_chain = metadata['receptor_chain']
        
        # Create protein objects only when needed
        protein_w_obj = Protein(
            pdb_filepath=pdbfile_raw, 
            read_water=True, 
            read_ligand=False, 
            read_chain=[receptor_chain] if receptor_chain else []
        )
        protein_obj = Protein(
            pdb_filepath=pdbfile, 
            read_water=False, 
            read_ligand=False
        )
        
        return protein_w_obj, protein_obj

    def _create_ligand_objects(self, metadata):
        """Create ligand objects from metadata"""
        ligfile = metadata['ligand_mol2']
        ligfile_for_vn = metadata['ligfile_for_vn']
        
        ligand_obj = Ligand(mol2_filepath=ligfile, drop_H=False)
        ligand_for_vn = None
        if ligfile_for_vn:
            ligand_for_vn = Ligand(mol2_filepath=ligfile_for_vn, drop_H=False)
            
        return ligand_obj, ligand_for_vn

    def _generate_graphs(self, metadata):
        """Generate all graphs for a sample"""
        try:
            # Create objects only when needed
            protein_w_obj, protein_obj = self._create_protein_objects(metadata)
            ligand_obj, ligand_for_vn = self._create_ligand_objects(metadata)
            
            center = metadata['center']
            name = metadata['name']
            
            # Initialize graphs
            graphs = {}
            
            # Generate protein-only graph (no water, no virtual nodes)
            protein_only_graph = self.protein_featurizer.featurize_no_water_graph(
                protein_obj, center=center, crop_size=self.crop_size
            )
            if protein_only_graph is None:
                print(f"[Warning] {name}: Failed to create protein-only graph")
                return None
            graphs['protein_only'] = protein_only_graph
            
            # Generate protein+water graph (water, no virtual nodes)
            protein_water_graph = self.protein_featurizer.featurize_graph_with_water(
                protein_w_obj, center=center, crop_size=self.crop_size
            )
            if protein_water_graph is None:
                print(f"[Warning] {name}: Failed to create protein+water graph")
                return None
            graphs['protein_water'] = protein_water_graph
            
            # Generate protein+water+virtual nodes graph (if not skipping virtual)
            if not self.skip_virtual:
                protein_virtual_graph = self.protein_featurizer.featurize_graph_with_virtual_nodes(
                    protein_w_water=protein_w_obj,
                    protein_wo_water=protein_obj,
                    ligand=ligand_obj,
                    ligand_for_vn=ligand_for_vn,
                    center=center,
                    crop_size=self.crop_size,
                    target_name=name
                )
                if protein_virtual_graph is None:
                    print(f"[Warning] {name}: Failed to create protein+virtual graph")
                    return None
                graphs['protein_virtual'] = protein_virtual_graph
            
            # Generate ligand graph
            ligand_graph = self.ligand_featurizer.featurize_graph(ligand_obj)
            if ligand_graph is None:
                print(f"[Warning] {name}: Failed to create ligand graph")
                return None
            graphs['ligand'] = ligand_graph
            
            # Create final sample
            sample = {
                'name': name,
                **graphs,
                'affinity': metadata['affinity'],
            }
            
            return sample
                
        except Exception as e:
            print(f"[Warning] Error generating graphs for {metadata['name']}: {e}")
            return None

    def __getitem__(self, idx):
        """Generate graphs lazily when requested"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Use modulo to wrap around if we've exhausted samples
                actual_idx = (idx + retry_count) % len(self.sample_metadata)
                metadata = self.sample_metadata[actual_idx]
                sample = self._generate_graphs(metadata)
                
                if sample is not None:
                    return sample
                else:
                    print(f"[Dataset] Sample {actual_idx} ({metadata['name']}) returned None, trying next sample...")
                    retry_count += 1
                    
            except Exception as e:
                print(f"[Dataset] Error in __getitem__ for index {actual_idx}: {e}")
                retry_count += 1
        
        # If all retries failed, return a dummy sample to prevent DataLoader crash
        print(f"[Dataset] All retries failed for index {idx}, returning dummy sample")
        return self._create_dummy_sample()

    def _create_dummy_sample(self):
        """Create a minimal dummy sample when all else fails"""
        # Create minimal dummy graphs to prevent DataLoader crashes
        # You may need to adjust these dimensions based on your actual feature sizes
        dummy_node_features = torch.zeros((1, 128), dtype=torch.float32)
        dummy_edge_index = torch.zeros((2, 0), dtype=torch.long)
        dummy_edge_features = torch.zeros((0, 64), dtype=torch.float32)
        
        dummy_graph = Data(
            x=dummy_node_features,
            edge_index=dummy_edge_index,
            edge_attr=dummy_edge_features
        )
        
        sample = {
            'name': 'dummy_sample',
            'protein_water': dummy_graph.clone(),
            'protein_only': dummy_graph.clone(),
            'ligand': dummy_graph.clone(),
            'affinity': torch.tensor(0.0, dtype=torch.float32),
        }
        
        if not self.skip_virtual:
            sample['protein_virtual'] = dummy_graph.clone()
            
        return sample

    def debug_sample(self, idx):
        """Debug a specific sample to see what's happening"""
        print(f"\n=== DEBUGGING SAMPLE {idx} ===")
        
        if idx >= len(self.sample_metadata):
            print(f"Index {idx} out of range. Dataset has {len(self.sample_metadata)} samples.")
            return None
            
        metadata = self.sample_metadata[idx]
        print(f"Sample name: {metadata['name']}")
        print(f"Metadata: {metadata}")
        
        # Check file existence
        files_to_check = ['pdb_file_biolip', 'pdb_file_db', 'ligand_mol2']
        for file_key in files_to_check:
            file_path = metadata.get(file_key)
            if file_path:
                exists = os.path.exists(file_path)
                print(f"{file_key}: {file_path} (exists: {exists})")
            else:
                print(f"{file_key}: None")
        
        if metadata.get('ligfile_for_vn'):
            exists = os.path.exists(metadata['ligfile_for_vn'])
            print(f"ligfile_for_vn: {metadata['ligfile_for_vn']} (exists: {exists})")
        
        # Try to create objects
        try:
            print("\n--- Creating protein objects ---")
            protein_w_obj, protein_obj = self._create_protein_objects(metadata)
            print(f"Protein with water: {protein_w_obj}")
            print(f"Protein without water: {protein_obj}")
        except Exception as e:
            print(f"Error creating protein objects: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        try:
            print("\n--- Creating ligand objects ---")
            ligand_obj, ligand_for_vn = self._create_ligand_objects(metadata)
            print(f"Main ligand: {ligand_obj}")
            print(f"VN ligand: {ligand_for_vn}")
        except Exception as e:
            print(f"Error creating ligand objects: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        try:
            print("\n--- Generating graphs ---")
            sample = self._generate_graphs(metadata)
            if sample:
                print("✓ Graphs generated successfully!")
                for key, value in sample.items():
                    if hasattr(value, 'x'):
                        print(f"{key}: nodes={value.x.shape[0]}, edges={value.edge_index.shape[1]}")
                    else:
                        print(f"{key}: {type(value)} = {value}")
                return sample
            else:
                print("✗ Graph generation returned None")
                return None
        except Exception as e:
            print(f"Error generating graphs: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_sample_info(self):
        """Get summary information about all samples"""
        print(f"\n=== DATASET INFO ===")
        print(f"Total samples: {len(self.sample_metadata)}")
        print(f"Mode: {self.mode}")
        print(f"Skip virtual: {self.skip_virtual}")
        print(f"Top K: {self.top_k}")
        print(f"Crop size: {self.crop_size}")
        
        if len(self.sample_metadata) > 0:
            print(f"\nFirst few samples:")
            for i, meta in enumerate(self.sample_metadata[:3]):
                print(f"  {i}: {meta['name']} (default: {meta['is_default']})")
        
        return {
            'total_samples': len(self.sample_metadata),
            'mode': self.mode,
            'skip_virtual': self.skip_virtual,
            'has_defaults': any(meta['is_default'] for meta in self.sample_metadata)
        }       

class RLADatasetLazy(Dataset):
    def __init__(self, data_path=None, target_dict: dict = None, mode='train', top_k=30, crop_size=30, validate_samples=True):
        self.data_path = data_path
        self.mode = mode
        self.top_k = top_k 
        self.crop_size = crop_size 
        self.skip_virtual = False
        self.validate_samples = validate_samples
        
        # Store only metadata for lazy loading
        self.sample_metadata = []
        
        # Instantiate featurizers (lightweight)
        self.protein_featurizer = ProteinFeaturizer(top_k=self.top_k)
        self.ligand_featurizer = LigandFeaturizer()

        # Default paths for testing mode
        self.default_protein_w_path = os.path.join(data_path, '2etr.pdb') if data_path else None
        self.default_protein_path = os.path.join(data_path, '2etr.pdb') if data_path else None
        self.default_ligand_path = os.path.join(data_path, '2etr_lig.mol2') if data_path else None

        if target_dict and len(target_dict) > 0:
            # Real data mode - process target_dict
            self._process_target_dict(target_dict)
        else:
            # Test/default mode - use default files
            self._create_default_metadata()
            
        print(f"[Dataset] Initialized {self.__class__.__name__} with {len(self.sample_metadata)} valid samples in {mode} mode")

    def _process_target_dict(self, target_dict):
        """Process real data from target_dict"""
        for target_info in target_dict:
            try:
                # Handle both JSON and CSV formats by checking available keys
                pdbfile = target_info.get('pdb_file_biolip', target_info.get('pdb_file', None))
                pdbfile_raw = target_info.get('pdb_file_db', target_info.get('pdb_file', None))
                receptor_chain = target_info.get('receptor_chain', None)
                ligfile = target_info.get('ligand_mol2', target_info.get('ligand_file', None))
                ligfile_for_vn = target_info.get('ligfile_for_vn', None)
                affinity_value = target_info.get('affinity', None)
                name = target_info.get('name', 'UnnamedTarget')
                center = target_info.get('center', None)

                # Check required fields
                if not pdbfile or not ligfile:
                    print(f"[Warning] Skipped {name}: Missing PDB or ligand file path.")
                    continue

                # Validate file existence if requested
                if self.validate_samples:
                    missing_files = []
                    if not os.path.exists(pdbfile):
                        missing_files.append(f"PDB: {pdbfile}")
                    if pdbfile_raw and pdbfile_raw != pdbfile and not os.path.exists(pdbfile_raw):
                        missing_files.append(f"Raw PDB: {pdbfile_raw}")
                    if not os.path.exists(ligfile):
                        missing_files.append(f"Ligand: {ligfile}")
                    if ligfile_for_vn and not os.path.exists(ligfile_for_vn):
                        missing_files.append(f"VN Ligand: {ligfile_for_vn}")
                    
                    if missing_files:
                        print(f"[Warning] Skipped {name}: Missing files - {', '.join(missing_files)}")
                        continue

                # Create metadata
                metadata = {
                    'name': name,
                    'pdb_file_biolip': pdbfile,
                    'pdb_file_db': pdbfile_raw if pdbfile_raw else pdbfile,
                    'receptor_chain': receptor_chain,
                    'ligand_mol2': ligfile,
                    'ligfile_for_vn': ligfile_for_vn,
                    'affinity': torch.tensor(float(affinity_value), dtype=torch.float32) if affinity_value is not None else None,
                    'center': torch.tensor(center, dtype=torch.float32) if center is not None else None,
                    'is_default': False
                }
                self.sample_metadata.append(metadata)
                
            except Exception as e:
                target_name_for_error = target_info.get('name', 'Unknown Target')
                print(f"[Warning] Skipped {target_name_for_error} due to error: {e}")

    def _create_default_metadata(self):
        """Create default metadata for testing"""
        if not self.data_path:
            print("[Warning] No data_path provided and no target_dict - cannot create default sample")
            return
            
        # Check if default files exist
        if self.validate_samples:
            missing_defaults = []
            if not os.path.exists(self.default_protein_path):
                missing_defaults.append(f"Default protein: {self.default_protein_path}")
            if not os.path.exists(self.default_protein_w_path):
                missing_defaults.append(f"Default protein with water: {self.default_protein_w_path}")
            if not os.path.exists(self.default_ligand_path):
                missing_defaults.append(f"Default ligand: {self.default_ligand_path}")
            
            if missing_defaults:
                print(f"[Warning] Cannot create default sample: {', '.join(missing_defaults)}")
                return

        # Create default metadata
        center = torch.tensor([50, 100, 30], dtype=torch.float32)
        default_metadata = {
            'name': 'default_target',
            'pdb_file_biolip': self.default_protein_path,
            'pdb_file_db': self.default_protein_w_path,
            'receptor_chain': 'A',
            'ligand_mol2': self.default_ligand_path,
            'ligfile_for_vn': None,
            'affinity': None,
            'center': center,
            'is_default': True
        }
        self.sample_metadata.append(default_metadata)

    def __len__(self):
        return len(self.sample_metadata)

    def _create_protein_objects(self, metadata):
        """Create protein objects from metadata"""
        pdbfile = metadata['pdb_file_biolip']
        pdbfile_raw = metadata['pdb_file_db']
        receptor_chain = metadata['receptor_chain']
        
        # Create protein objects only when needed
        protein_w_obj = Protein(
            pdb_filepath=pdbfile_raw, 
            read_water=True, 
            read_ligand=False, 
            read_chain=[receptor_chain] if receptor_chain else []
        )
        protein_obj = Protein(
            pdb_filepath=pdbfile, 
            read_water=False, 
            read_ligand=False
        )
        
        return protein_w_obj, protein_obj

    def _create_ligand_objects(self, metadata):
        """Create ligand objects from metadata"""
        ligfile = metadata['ligand_mol2']
        ligfile_for_vn = metadata['ligfile_for_vn']
        
        ligand_obj = Ligand(mol2_filepath=ligfile, drop_H=False)
        ligand_for_vn = None
        if ligfile_for_vn:
            ligand_for_vn = Ligand(mol2_filepath=ligfile_for_vn, drop_H=False)
            
        return ligand_obj, ligand_for_vn

    def _generate_graphs(self, metadata):
        """Generate all graphs for a sample"""
        try:
            # Create objects only when needed
            protein_w_obj, protein_obj = self._create_protein_objects(metadata)
            ligand_obj, ligand_for_vn = self._create_ligand_objects(metadata)
            
            center = metadata['center']
            name = metadata['name']
            
            # Initialize graphs
            graphs = {}
            
            # Generate protein-only graph (no water, no virtual nodes)
            protein_only_graph = self.protein_featurizer.featurize_no_water_graph(
                protein_obj, center=center, crop_size=self.crop_size
            )
            if protein_only_graph is None:
                print(f"[Warning] {name}: Failed to create protein-only graph")
                return None
            graphs['protein_only'] = protein_only_graph
            
            # Generate protein+water graph (water, no virtual nodes)
            protein_water_graph = self.protein_featurizer.featurize_graph_with_water(
                protein_w_obj, center=center, crop_size=self.crop_size
            )
            if protein_water_graph is None:
                print(f"[Warning] {name}: Failed to create protein+water graph")
                return None
            graphs['protein_water'] = protein_water_graph
            
            # Generate protein+water+virtual nodes graph (if not skipping virtual)
            if not self.skip_virtual:
                protein_virtual_graph = self.protein_featurizer.featurize_graph_with_virtual_nodes(
                    protein_w_water=protein_w_obj,
                    protein_wo_water=protein_obj,
                    ligand=ligand_obj,
                    ligand_for_vn=ligand_for_vn,
                    center=center,
                    crop_size=self.crop_size,
                    target_name=name
                )
                if protein_virtual_graph is None:
                    print(f"[Warning] {name}: Failed to create protein+virtual graph")
                    return None
                graphs['protein_virtual'] = protein_virtual_graph
            
            # Generate ligand graph
            ligand_graph = self.ligand_featurizer.featurize_graph(ligand_obj)
            if ligand_graph is None:
                print(f"[Warning] {name}: Failed to create ligand graph")
                return None
            graphs['ligand'] = ligand_graph
            
            # Create final sample
            sample = {
                'name': name,
                **graphs,
                'affinity': metadata['affinity'],
            }
            
            return sample
                
        except Exception as e:
            print(f"[Warning] Error generating graphs for {metadata['name']}: {e}")
            return None

    def __getitem__(self, idx):
        """Generate graphs lazily when requested"""
        if len(self.sample_metadata) == 0:
            raise RuntimeError("Dataset has no valid samples!")
        
        max_retries = len(self.sample_metadata)  # Don't retry more than total samples
        retry_count = 0
        attempted_indices = set()
        
        while retry_count < max_retries:
            try:
                # Use modulo to wrap around if we've exhausted samples
                actual_idx = (idx + retry_count) % len(self.sample_metadata)
                
                # Avoid infinite loops by tracking attempted indices
                if actual_idx in attempted_indices:
                    retry_count += 1
                    continue
                attempted_indices.add(actual_idx)
                
                metadata = self.sample_metadata[actual_idx]
                sample = self._generate_graphs(metadata)
                
                if sample is not None:
                    return sample
                else:
                    print(f"[Dataset] Sample {actual_idx} ({metadata['name']}) returned None, trying next sample...")
                    retry_count += 1
                    
            except Exception as e:
                print(f"[Dataset] Error in __getitem__ for index {actual_idx}: {e}")
                retry_count += 1
        
        # If all retries failed, this indicates a serious problem
        print(f"[Dataset] CRITICAL: All {max_retries} retries failed for index {idx}")
        print(f"[Dataset] Attempted indices: {sorted(attempted_indices)}")
        print(f"[Dataset] This suggests a fundamental issue with your data or featurizers")
        
        # Return dummy sample as last resort, but this indicates a problem
        return self._create_dummy_sample()

    def _create_dummy_sample(self):
        """Create a minimal dummy sample when all else fails"""
        print("[Warning] Creating dummy sample - this indicates a problem with your data")
        
        # Try to get feature dimensions from a working sample first
        dummy_node_features = None
        dummy_edge_features = None
        
        # Look for a working sample to get correct dimensions
        for i in range(min(5, len(self.sample_metadata))):
            try:
                metadata = self.sample_metadata[i]
                sample = self._generate_graphs(metadata)
                if sample and 'protein_only' in sample:
                    ref_graph = sample['protein_only']
                    if hasattr(ref_graph, 'x') and ref_graph.x is not None:
                        node_dim = ref_graph.x.shape[1]
                        dummy_node_features = torch.zeros((1, node_dim), dtype=torch.float32)
                        print(f"[Dummy] Using node feature dim: {node_dim}")
                    
                    if hasattr(ref_graph, 'edge_attr') and ref_graph.edge_attr is not None and ref_graph.edge_attr.numel() > 0:
                        edge_dim = ref_graph.edge_attr.shape[1]
                        dummy_edge_features = torch.zeros((0, edge_dim), dtype=torch.float32)
                        print(f"[Dummy] Using edge feature dim: {edge_dim}")
                    else:
                        dummy_edge_features = None
                        print(f"[Dummy] No edge features found")
                    break
            except:
                continue
        
        # Fallback dimensions if we couldn't find any working sample
        if dummy_node_features is None:
            print("[Warning] Could not determine node feature dimensions, using fallback")
            dummy_node_features = torch.zeros((1, 20), dtype=torch.float32)  # Conservative guess
        
        if dummy_edge_features is None:
            print("[Warning] Using no edge features for dummy sample")
            dummy_edge_features = torch.empty((0, 0), dtype=torch.float32)
        
        dummy_edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        dummy_graph = Data(
            x=dummy_node_features,
            edge_index=dummy_edge_index,
            edge_attr=dummy_edge_features if dummy_edge_features.numel() > 0 else None
        )
        
        sample = {
            'name': 'dummy_sample',
            'protein_water': dummy_graph.clone(),
            'protein_only': dummy_graph.clone(),
            'ligand': dummy_graph.clone(),
            'affinity': torch.tensor(0.0, dtype=torch.float32),
        }
        
        if not self.skip_virtual:
            sample['protein_virtual'] = dummy_graph.clone()
            
        return sample

    def debug_sample(self, idx):
        """Debug a specific sample to see what's happening"""
        print(f"\n=== DEBUGGING SAMPLE {idx} ===")
        
        if idx >= len(self.sample_metadata):
            print(f"Index {idx} out of range. Dataset has {len(self.sample_metadata)} samples.")
            return None
            
        metadata = self.sample_metadata[idx]
        print(f"Sample name: {metadata['name']}")
        print(f"Metadata: {metadata}")
        
        # Check file existence
        files_to_check = ['pdb_file_biolip', 'pdb_file_db', 'ligand_mol2']
        for file_key in files_to_check:
            file_path = metadata.get(file_key)
            if file_path:
                exists = os.path.exists(file_path)
                print(f"{file_key}: {file_path} (exists: {exists})")
            else:
                print(f"{file_key}: None")
        
        if metadata.get('ligfile_for_vn'):
            exists = os.path.exists(metadata['ligfile_for_vn'])
            print(f"ligfile_for_vn: {metadata['ligfile_for_vn']} (exists: {exists})")
        
        # Try to create objects
        try:
            print("\n--- Creating protein objects ---")
            protein_w_obj, protein_obj = self._create_protein_objects(metadata)
            print(f"Protein with water: {protein_w_obj}")
            print(f"Protein without water: {protein_obj}")
        except Exception as e:
            print(f"Error creating protein objects: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        try:
            print("\n--- Creating ligand objects ---")
            ligand_obj, ligand_for_vn = self._create_ligand_objects(metadata)
            print(f"Main ligand: {ligand_obj}")
            print(f"VN ligand: {ligand_for_vn}")
        except Exception as e:
            print(f"Error creating ligand objects: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        try:
            print("\n--- Generating graphs ---")
            sample = self._generate_graphs(metadata)
            if sample:
                print("✓ Graphs generated successfully!")
                for key, value in sample.items():
                    if hasattr(value, 'x'):
                        print(f"{key}: nodes={value.x.shape[0]}, edges={value.edge_index.shape[1]}")
                    else:
                        print(f"{key}: {type(value)} = {value}")
                return sample
            else:
                print("✗ Graph generation returned None")
                return None
        except Exception as e:
            print(f"Error generating graphs: {e}")
            import traceback
            traceback.print_exc()
            return None

    def validate_feature_consistency(self, num_samples=5):
        """Validate that samples have consistent feature dimensions"""
        print(f"\n=== VALIDATING FEATURE CONSISTENCY ===")
        
        feature_dims = {}
        valid_samples = 0
        
        for i in range(min(num_samples, len(self.sample_metadata))):
            try:
                print(f"\nChecking sample {i}: {self.sample_metadata[i]['name']}")
                sample = self._generate_graphs(self.sample_metadata[i])
                
                if sample is None:
                    print(f"  Sample {i}: Failed to generate")
                    continue
                
                valid_samples += 1
                
                for key, graph in sample.items():
                    if hasattr(graph, 'x') and graph.x is not None:
                        node_dim = graph.x.shape[1]
                        if key not in feature_dims:
                            feature_dims[key] = {'node_dims': set(), 'edge_dims': set()}
                        feature_dims[key]['node_dims'].add(node_dim)
                        
                        edge_dim = None
                        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None and graph.edge_attr.numel() > 0:
                            edge_dim = graph.edge_attr.shape[1]
                            feature_dims[key]['edge_dims'].add(edge_dim)
                        
                        print(f"  {key}: nodes={graph.x.shape[0]}x{node_dim}, edges={graph.edge_index.shape[1]}x{edge_dim}")
                        
            except Exception as e:
                print(f"  Sample {i}: Error - {e}")
        
        print(f"\n=== CONSISTENCY REPORT ===")
        print(f"Valid samples checked: {valid_samples}/{num_samples}")
        
        all_consistent = True
        for key, dims in feature_dims.items():
            node_dims = dims['node_dims']
            edge_dims = dims['edge_dims']
            
            node_consistent = len(node_dims) <= 1
            edge_consistent = len(edge_dims) <= 1
            
            print(f"\n{key}:")
            print(f"  Node feature dims: {sorted(node_dims)} ({'✓ Consistent' if node_consistent else '✗ INCONSISTENT'})")
            print(f"  Edge feature dims: {sorted(edge_dims) if edge_dims else 'None'} ({'✓ Consistent' if edge_consistent else '✗ INCONSISTENT'})")
            
            if not (node_consistent and edge_consistent):
                all_consistent = False
        
        if all_consistent:
            print(f"\n✓ All features are consistent!")
        else:
            print(f"\n✗ Feature inconsistencies detected!")
            
        return feature_dims, all_consistent

    def get_sample_info(self):
        """Get summary information about all samples"""
        print(f"\n=== DATASET INFO ===")
        print(f"Total samples: {len(self.sample_metadata)}")
        print(f"Mode: {self.mode}")
        print(f"Skip virtual: {self.skip_virtual}")
        print(f"Top K: {self.top_k}")
        print(f"Crop size: {self.crop_size}")
        
        if len(self.sample_metadata) > 0:
            print(f"\nFirst few samples:")
            for i, meta in enumerate(self.sample_metadata[:3]):
                print(f"  {i}: {meta['name']} (default: {meta['is_default']})")
        
        return {
            'total_samples': len(self.sample_metadata),
            'mode': self.mode,
            'skip_virtual': self.skip_virtual,
            'has_defaults': any(meta['is_default'] for meta in self.sample_metadata)
        }
     
if __name__ == '__main__':
    # Example usage
    ProteinDataset = RLADataset(data_path='data', mode='train')
