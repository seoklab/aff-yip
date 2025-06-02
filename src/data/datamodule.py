import pandas as pd
import numpy as np
import json
import os
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as PyGDataLoader

from .dataset_gvp import RLADataset # Or appropriate import

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
        file_path_keys_from_json = ['pdb_file_biolip', 'pdb_file_db', 'ligand_pdb', 'ligand_mol2']
        
        # Define all keys considered essential for an entry to be valid.
        # If any of these are missing, the entry is skipped.
        essential_keys_from_json = [
            'name', 'pdb_file_biolip', 'pdb_file_db', 
            'ligand_pdb', 'ligand_mol2', 
            'affinity', 'receptor_chain', 'center' 
        ] # 'err_flag' is checked separately

        for index, item_in_json in enumerate(raw_meta_list):
            # 1. Check for err_flag first
            if item_in_json.get('err_flag', False):
                print(f"[Info] Skipping item {item_in_json.get('name', index)} from '{json_filepath}' due to err_flag=True.")
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
        return PyGDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False)
    def val_dataloader(self):
        if not self.val_dataset: print("[Warning] Validation dataset not available."); return None
        return PyGDataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False)
    def test_dataloader(self):
        if not self.test_dataset: print("[Warning] Test dataset not available."); return None
        return PyGDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False)

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
