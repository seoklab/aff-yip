import pandas as pd
import numpy as np
import json
import os
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch
import torch_geometric
import torch 

from .dataset_gvp import RLADataset # Or appropriate import

def _log_bad_samples(names, path='bad_samples.txt'):
    """Append bad sample names to a file"""
    try:
        with open(path, 'a') as f:
            for name in names:
                f.write(f"{name}\n")
    except Exception as log_err:
        print(f"[Logger] Failed to write to {path}: {log_err}")

class RLADataModule(pl.LightningDataModule):
    def __init__(self,
                 data_path: str = None, # Root path for actual PDB/ligand files, used by RLADataset
                 train_meta_path: str = None,
                 val_meta_path: str = None,
                 test_meta_path: str = None,
                 batch_size: int = 1,
                 num_workers: int = 4,
                 top_k: int = 30,
                 crop_size: int = 30): # Removed json_pdb_key, json_ligand_key
        super().__init__()
        self.data_path = data_path
        self.train_meta_path = train_meta_path
        self.val_meta_path = val_meta_path
        self.test_meta_path = test_meta_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.top_k = top_k
        self.crop_size = crop_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    @staticmethod
    def _load_meta_from_csv(csv_filepath, data_root_path=""):
        try:
            df = pd.read_csv(csv_filepath)
        except FileNotFoundError:
            print(f"[Error] CSV file not found: {csv_filepath}")
            return []
        meta_list = []
        # These are the keys RLADataset would look for if processing a CSV-derived dict
        # If your CSV has different names, you'd map them here.
        csv_pdb_col = 'pdb_file'
        csv_ligand_col = 'ligand_file'
        required_cols = [csv_pdb_col, csv_ligand_col]

        if not all(col in df.columns for col in required_cols):
            print(f"[Error] CSV '{csv_filepath}' must contain at least columns: {', '.join(required_cols)}")
            return []

        for index, row in df.iterrows():
            pdb_file = row[csv_pdb_col]
            ligand_file = row[csv_ligand_col]

            if data_root_path and not os.path.isabs(pdb_file):
                pdb_file = os.path.join(data_root_path, pdb_file)
            if data_root_path and not os.path.isabs(ligand_file):
                ligand_file = os.path.join(data_root_path, ligand_file)

            sample_meta = {
                'name': row.get('name', f'sample_{index}'),
                'pdb_file': pdb_file, # This key is for RLADataset
                'ligand_file': ligand_file, # This key is for RLADataset
                'affinity': float(row.get('affinity', np.nan))
            }
            # Handle center for CSV as before
            if 'center_x' in row and 'center_y' in row and 'center_z' in row:
                try:
                    center = [float(row['center_x']), float(row['center_y']), float(row['center_z'])]
                    sample_meta['center'] = center
                except ValueError: print(f"[Warning] Could not parse center for CSV row {index}.")
            elif 'center_str' in row:
                try:
                    import ast
                    parsed_center = ast.literal_eval(row['center_str'])
                    if isinstance(parsed_center, list) and len(parsed_center) == 3 and all(isinstance(c, (int, float)) for c in parsed_center):
                        sample_meta['center'] = parsed_center
                    else: print(f"[Warning] Parsed 'center_str' for CSV row {index} invalid.")
                except (ValueError, SyntaxError): print(f"[Warning] Could not parse 'center_str' for CSV row {index}.")
            meta_list.append(sample_meta)
        return meta_list


    @staticmethod
    def _load_meta_from_json(json_filepath, data_root_path=""):
        try:
            with open(json_filepath, 'r') as f:
                raw_meta_list = json.load(f)
        except FileNotFoundError:
            print(f"[Error] JSON file not found: {json_filepath}")
            return []
        except json.JSONDecodeError:
            print(f"[Error] Could not decode JSON from file: {json_filepath}")
            return []

        if not isinstance(raw_meta_list, list):
            print(f"[Error] JSON file '{json_filepath}' should contain a list of objects.")
            return []

        processed_meta_list = []
        
        # Define all file path keys from your JSON structure that need path joining
        file_path_keys_from_json = ['pdb_file_biolip', 'pdb_file_db', 'ligand_mol2', 'ligfile_for_vn']
        
        # Define all keys considered essential for an entry to be valid.
        # If any of these are missing, the entry is skipped.
        essential_keys_from_json = [
            'name', 'pdb_file_biolip', 'pdb_file_db', 
            'ligand_mol2', 'ligfile_for_vn', 
            'affinity', 'receptor_chain', 'center' 
        ] # 'err_flag' is checked separately

        for index, item_in_json in enumerate(raw_meta_list):
            # 1. Check for err_flag first
            if item_in_json.get('err_flag', False):
                # print(f"[Info] Skipping item {item_in_json.get('name', index)} from '{json_filepath}' due to err_flag=True.")
                continue
            name = item_in_json.get('name', '')
            if 'rna' in name.lower():
                # print(f"[Info] Skipping item {name} because name contains 'rna'.")
                continue
            if 'dna' in name.lower():
                # print(f"[Info] Skipping item {name} because name contains 'dna'.")
                continue
            if name.islower():
                # print(f"[Info] Skipping item {name} because name is all lower case.")
                continue
                   

            # 2. Check for presence of all essential keys
            missing_keys = [k for k in essential_keys_from_json if k not in item_in_json]
            if missing_keys:
                print(f"[Warning] Item {item_in_json.get('name', index)} in '{json_filepath}' is missing essential keys: {missing_keys}. Skipping.")
                continue
            
            # 3. Create the item for target_dict, processing paths and types as needed
            processed_target_item = {}
            valid_item = True # Flag to track if current item processing is successful

            for key, value in item_in_json.items():
                if key in file_path_keys_from_json:
                    path_value = value
                    if not isinstance(path_value, str): # Basic check for path type
                        print(f"[Warning] Path value for key '{key}' in item {item_in_json.get('name', index)} is not a string: {path_value}. Skipping item.")
                        valid_item = False; break
                    if data_root_path and not os.path.isabs(path_value):
                        path_value = os.path.join(data_root_path, path_value)
                    processed_target_item[key] = path_value
                elif key == 'affinity':
                    try:
                        processed_target_item[key] = float(value) 
                    except (ValueError, TypeError):
                        print(f"[Warning] Could not convert affinity '{value}' to float for item {item_in_json.get('name', index)}. Skipping item.")
                        valid_item = False; break
                elif key == 'center':
                    if isinstance(value, list) and len(value) == 3 and all(isinstance(c, (int, float)) for c in value):
                        processed_target_item[key] = value
                    else:
                        print(f"[Warning] 'center' for item {item_in_json.get('name', index)} is not a valid list of 3 numbers. Skipping item as 'center' is essential.")
                        valid_item = False; break # Assuming center is essential as per essential_keys_from_json
                else:
                    # For other keys like 'name', 'receptor_chain', 'affinity_type', 'err_flag' (which is False here)
                    processed_target_item[key] = value
            
            if not valid_item: # If item was marked invalid during processing
                continue

            processed_meta_list.append(processed_target_item)
            
        return processed_meta_list

    def _load_meta_file(self, meta_filepath: str):
        if not meta_filepath:
            return None
        if not os.path.exists(meta_filepath):
            print(f"[Warning] Metadata file not found: {meta_filepath}")
            return None
            
        _, ext = os.path.splitext(meta_filepath)
        if ext == '.csv':
            return self._load_meta_from_csv(meta_filepath, self.data_path)
        elif ext == '.json':
            # No longer pass pdb_key_in_json or ligand_key_in_json
            return self._load_meta_from_json(meta_filepath, self.data_path)
        else:
            raise ValueError(f"Unsupported metadata file format: {ext} for {meta_filepath}. Please use .csv or .json.")

    @staticmethod
    def safe_collate_fn(batch, error_log_path='bad_samples.txt'):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        collated = {}
        for key in batch[0].keys():
            values = [sample[key] for sample in batch]

            if isinstance(values[0], torch_geometric.data.Data):
                try:
                    collated[key] = Batch.from_data_list(values)
                except Exception as e:
                    names = [getattr(d, 'name', f"<no_name_{i}>") for i, d in enumerate(values)]
                    print(f"[Collate] Skipping batch for key '{key}' due to DataList error: {e}")
                    print(f"         Affected samples: {names}")
                    _log_bad_samples(names, error_log_path)
                    return None
            elif isinstance(values[0], str):
                collated[key] = values
            else:
                try:
                    collated[key] = torch.tensor(values)
                except Exception as e:
                    print(f"[Collate] Skipping batch for key '{key}' due to tensor error: {e}")
                    print(f"         Values: {values}")
                    names = [getattr(sample['ligand'], 'name', f"<no_name_{i}>") for i, sample in batch if 'ligand' in sample]
                    print(f"         Affected sample names: {names}")
                    _log_bad_samples(names, error_log_path)
                    return None
        return collated
        
    def setup(self, stage: str = None):
        # ... (setup logic remains the same, it will use the loaded targets)
        if stage == 'fit' or stage is None:
            if self.train_meta_path:
                train_targets = self._load_meta_file(self.train_meta_path)
                if train_targets: 
                    self.train_dataset = RLADataset(
                        data_path=self.data_path, target_dict=train_targets, mode='train',
                        top_k=self.top_k, crop_size=self.crop_size
                    )
            if self.val_meta_path:
                val_targets = self._load_meta_file(self.val_meta_path)
                if val_targets:
                    self.val_dataset = RLADataset(
                        data_path=self.data_path, target_dict=val_targets, mode='val',
                        top_k=self.top_k, crop_size=self.crop_size
                    )
        if stage == 'test' or stage is None:
            if self.test_meta_path:
                test_targets = self._load_meta_file(self.test_meta_path)
                if test_targets:
                    self.test_dataset = RLADataset(
                        data_path=self.data_path, target_dict=test_targets, mode='test',
                        top_k=self.top_k, crop_size=self.crop_size
                    )

    # Dataloader methods (train_dataloader, val_dataloader, test_dataloader) remain the same
    def train_dataloader(self):
        if not self.train_dataset: print("[Warning] Training dataset not available."); return None 
        return PyGDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=RLADataModule.safe_collate_fn, persistent_workers=True if self.num_workers > 0 else False)
    def val_dataloader(self):
        if not self.val_dataset: print("[Warning] Validation dataset not available."); return None
        return PyGDataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=RLADataModule.safe_collate_fn, persistent_workers=True if self.num_workers > 0 else False)
    def test_dataloader(self):
        if not self.test_dataset: print("[Warning] Test dataset not available."); return None
        return PyGDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=RLADataModule.safe_collate_fn, persistent_workers=True if self.num_workers > 0 else False)

class DebugRLADataModule(pl.LightningDataModule):
    def __init__(self, 
                 original_datamodule: 'RLADataModule', 
                 debug_samples: int = 3,
                 debug_batch_size: int = 1,
                 debug_num_workers: int = 0):
        """
        Debug wrapper for RLADataModule that limits to first few samples for fast debugging.
        
        Args:
            original_datamodule: The original RLADataModule instance
            debug_samples: Number of samples to use for debugging (default: 3)
            debug_batch_size: Batch size for debugging (default: 1)
            debug_num_workers: Number of workers for debugging (default: 0, no multiprocessing)
        """
        super().__init__()
        self.original_dm = original_datamodule
        self.debug_samples = debug_samples
        self.debug_batch_size = debug_batch_size
        self.debug_num_workers = debug_num_workers
        
        # Copy essential parameters from original
        self.data_path = original_datamodule.data_path
        self.train_meta_path = original_datamodule.train_meta_path
        self.val_meta_path = original_datamodule.val_meta_path
        self.test_meta_path = original_datamodule.test_meta_path
        self.top_k = original_datamodule.top_k
        self.crop_size = original_datamodule.crop_size
        
        # Will be populated in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _limit_meta_list(self, meta_list, max_samples=None):
        """Limit meta list to first few samples for debugging"""
        if meta_list is None:
            return None
        
        max_samples = max_samples or self.debug_samples
        limited_list = meta_list[:max_samples]
        
        print(f"[Debug] Limited dataset from {len(meta_list)} to {len(limited_list)} samples")
        
        # Print sample info for debugging
        for i, sample in enumerate(limited_list):
            print(f"[Debug] Sample {i}: {sample.get('name', 'unnamed')}")
            if 'center' in sample:
                print(f"         Center: {sample['center']}")
            if 'affinity' in sample:
                print(f"         Affinity: {sample['affinity']}")
        
        return limited_list

    def setup(self, stage: str = None):
        """Setup debug datasets using limited samples"""
        print(f"\n=== DEBUG DATAMODULE SETUP (stage: {stage}) ===")
        print(f"Debug samples: {self.debug_samples}")
        print(f"Debug batch size: {self.debug_batch_size}")
        
        if stage == 'fit' or stage is None:
            # Setup training dataset
            if self.train_meta_path:
                print(f"[Debug] Loading training metadata from: {self.train_meta_path}")
                train_targets = self.original_dm._load_meta_file(self.train_meta_path)
                train_targets_limited = self._limit_meta_list(train_targets)
                
                if train_targets_limited:
                    print(f"[Debug] Creating training dataset with {len(train_targets_limited)} samples")
                    self.train_dataset = RLADataset(
                        data_path=self.data_path, 
                        target_dict=train_targets_limited, 
                        mode='train',
                        top_k=self.top_k, 
                        crop_size=self.crop_size
                    )
                    print(f"[Debug] Training dataset created with {len(self.train_dataset)} samples")
                else:
                    print("[Debug] No training targets available")
            
            # Setup validation dataset
            if self.val_meta_path:
                print(f"[Debug] Loading validation metadata from: {self.val_meta_path}")
                val_targets = self.original_dm._load_meta_file(self.val_meta_path)
                val_targets_limited = self._limit_meta_list(val_targets, max_samples=min(2, self.debug_samples))
                
                if val_targets_limited:
                    print(f"[Debug] Creating validation dataset with {len(val_targets_limited)} samples")
                    self.val_dataset = RLADataset(
                        data_path=self.data_path, 
                        target_dict=val_targets_limited, 
                        mode='val',
                        top_k=self.top_k, 
                        crop_size=self.crop_size
                    )
                    print(f"[Debug] Validation dataset created with {len(self.val_dataset)} samples")
                else:
                    print("[Debug] No validation targets available")
        
        if stage == 'test' or stage is None:
            # Setup test dataset
            if self.test_meta_path:
                print(f"[Debug] Loading test metadata from: {self.test_meta_path}")
                test_targets = self.original_dm._load_meta_file(self.test_meta_path)
                test_targets_limited = self._limit_meta_list(test_targets, max_samples=1)
                
                if test_targets_limited:
                    print(f"[Debug] Creating test dataset with {len(test_targets_limited)} samples")
                    self.test_dataset = RLADataset(
                        data_path=self.data_path, 
                        target_dict=test_targets_limited, 
                        mode='test',
                        top_k=self.top_k, 
                        crop_size=self.crop_size
                    )
                    print(f"[Debug] Test dataset created with {len(self.test_dataset)} samples")
                else:
                    print("[Debug] No test targets available")

    def train_dataloader(self):
        if not self.train_dataset:
            print("[Debug Warning] Training dataset not available.")
            return None
        
        print(f"[Debug] Creating train dataloader with batch_size={self.debug_batch_size}, num_workers={self.debug_num_workers}")
        return PyGDataLoader(
            self.train_dataset, 
            batch_size=self.debug_batch_size, 
            shuffle=False,  # Don't shuffle for debugging consistency
            num_workers=self.debug_num_workers, 
            collate_fn=RLADataModule.safe_collate_fn, 
            persistent_workers=False  # Always False for debugging
        )

    def val_dataloader(self):
        if not self.val_dataset:
            print("[Debug Warning] Validation dataset not available.")
            return None
        
        print(f"[Debug] Creating val dataloader with batch_size={self.debug_batch_size}, num_workers={self.debug_num_workers}")
        return PyGDataLoader(
            self.val_dataset, 
            batch_size=self.debug_batch_size, 
            shuffle=False, 
            num_workers=self.debug_num_workers, 
            collate_fn=RLADataModule.safe_collate_fn, 
            persistent_workers=False
        )

    def test_dataloader(self):
        if not self.test_dataset:
            print("[Debug Warning] Test dataset not available.")
            return None
        
        print(f"[Debug] Creating test dataloader with batch_size={self.debug_batch_size}, num_workers={self.debug_num_workers}")
        return PyGDataLoader(
            self.test_dataset, 
            batch_size=self.debug_batch_size, 
            shuffle=False, 
            num_workers=self.debug_num_workers, 
            collate_fn=RLADataModule.safe_collate_fn, 
            persistent_workers=False
        )

    def debug_single_batch(self, stage='train'):
        """Get a single batch for debugging purposes"""
        self.setup(stage='fit' if stage in ['train', 'val'] else 'test')
        
        if stage == 'train' and self.train_dataset:
            loader = self.train_dataloader()
        elif stage == 'val' and self.val_dataset:
            loader = self.val_dataloader()
        elif stage == 'test' and self.test_dataset:
            loader = self.test_dataloader()
        else:
            print(f"[Debug Error] No dataset available for stage: {stage}")
            return None
        
        if loader is None:
            print(f"[Debug Error] No dataloader available for stage: {stage}")
            return None
        
        try:
            batch = next(iter(loader))
            print(f"[Debug] Successfully got batch for stage: {stage}")
            return batch
        except Exception as e:
            print(f"[Debug Error] Failed to get batch for stage {stage}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def print_debug_info(self):
        """Print comprehensive debug information"""
        print("\n" + "="*50)
        print("DEBUG DATAMODULE INFORMATION")
        print("="*50)
        print(f"Original DataModule: {type(self.original_dm).__name__}")
        print(f"Debug samples: {self.debug_samples}")
        print(f"Debug batch size: {self.debug_batch_size}")
        print(f"Debug num workers: {self.debug_num_workers}")
        print(f"Data path: {self.data_path}")
        print(f"Train meta path: {self.train_meta_path}")
        print(f"Val meta path: {self.val_meta_path}")
        print(f"Test meta path: {self.test_meta_path}")
        print(f"Top K: {self.top_k}")
        print(f"Crop size: {self.crop_size}")
        
        if hasattr(self, 'train_dataset') and self.train_dataset:
            print(f"Train dataset size: {len(self.train_dataset)}")
        if hasattr(self, 'val_dataset') and self.val_dataset:
            print(f"Val dataset size: {len(self.val_dataset)}")
        if hasattr(self, 'test_dataset') and self.test_dataset:
            print(f"Test dataset size: {len(self.test_dataset)}")
        print("="*50)



if __name__ == "__main__":
        dummy_json_path = "/home/j2ho/DB/biolip/test.json"
        print("\n--- Testing RLADataModule ---")
        try:
            data_module = RLADataModule(
                data_path=".", # Use parent directory of dummy_json_path
                train_meta_path=str(dummy_json_path),
                val_meta_path=str(dummy_json_path), # Using same for val/test for simplicity
                test_meta_path=str(dummy_json_path),
                batch_size=1, # Small batch size for testing
                num_workers=0, # Usually 0 for easier debugging, set to >0 to test multiprocessing
                top_k=30,      # Example value
                crop_size=20   # Example value
            )

            print("\n--- Calling data_module.setup('fit')... ---")
            data_module.setup('fit') # Load train and val datasets

            print("\n--- Checking training dataloader... ---")
            train_loader = data_module.train_dataloader()
            if train_loader:
                print(f"Train dataset size: {len(data_module.train_dataset) if data_module.train_dataset else 'N/A'}")
                if data_module.train_dataset and len(data_module.train_dataset) > 0:
                    print("Fetching one batch from train_loader...")
                    try:
                        batch = next(iter(train_loader))
                        print("Successfully fetched a batch from train_loader.")
                        print(f"Batch type: {type(batch)}")
                        if hasattr(batch, 'keys'): # For PyG Batch object which is like a dict
                             print(f"Keys in batch object: {batch.keys}")
                        # You can add more detailed checks on the batch content here
                        # For example, if 'protein_virtual' is a key from RLADataset output:
                        # if 'protein_virtual' in batch:
                        #    print(f"protein_virtual graph in batch: {batch['protein_virtual']}")
                        # else:
                        #    print(f"Sample batch structure: {batch}")
                    except Exception as e:
                        print(f"Error fetching/processing batch from train_loader: {e}")
                        print("This might indicate an issue within RLADataset or its featurizers with the dummy data.")
                else:
                    print("Train dataset is empty or None. Cannot fetch batch.")
            else:
                print("Train loader is None.")

            print("\n--- Calling data_module.setup('test')... ---")
            data_module.setup('test') # Load test dataset

            print("\n--- Checking test dataloader... ---")
            test_loader = data_module.test_dataloader()
            if test_loader:
                print(f"Test dataset size: {len(data_module.test_dataset) if data_module.test_dataset else 'N/A'}")
                if data_module.test_dataset and len(data_module.test_dataset) > 0:
                    print("Test loader created.")
                else:
                    print("Test dataset is empty or None.")

            else:
                print("Test loader is None.")

            print("\n--- Test block finished ---")

        except Exception as e:
            print(f"\nAn error occurred during RLADataModule testing: {e}")
            import traceback
            traceback.print_exc()
