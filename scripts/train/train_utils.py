# Updated coordinate saving utilities for sidechain_map format

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np

class CoordinateSaverCallback(Callback):
    """
    Updated callback for saving coordinates using sidechain_map format.
    Handles the new EGNN output format and fixes inference mode issues.
    """
    def __init__(self,
                 save_every_n_epochs=20,
                 original_mol2_dir="/home/j2ho/DB/biolip/BioLiP_updated_set/ligand_mol2",
                 original_pdb_dir="/home/j2ho/DB/biolip/BioLiP_updated_set/receptor",
                 output_dir="./predictions",
                 save_coords_pt=True,
                 separate_epoch_dirs=False,  # Default to False now
                 include_epoch_in_filename=True):
        super().__init__()
        self.save_every_n_epochs = save_every_n_epochs
        self.original_mol2_dir = original_mol2_dir
        self.original_pdb_dir = original_pdb_dir
        self.output_dir = output_dir
        self.save_coords_pt = save_coords_pt
        self.separate_epoch_dirs = separate_epoch_dirs
        self.include_epoch_in_filename = include_epoch_in_filename

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.save_every_n_epochs == 0:
            if pl_module.hparams.predict_str:
                print(f"\nSaving coordinates for epoch {trainer.current_epoch + 1}...")
                self.save_coordinates(trainer, pl_module, stage='val')

    def on_test_end(self, trainer, pl_module):
        if pl_module.hparams.predict_str:
            print("\nSaving final test coordinates...")
            self.save_coordinates(trainer, pl_module, stage='test')

    def save_coordinates(self, trainer, pl_module, stage='val'):
        """
        Main method to save coordinates using the new sidechain_map format.
        """
        if stage == 'val':
            dataloader = trainer.val_dataloaders
        elif stage == 'test':
            dataloader = trainer.test_dataloaders
        else:
            return

        # Set model to eval mode but keep gradients enabled to avoid inference mode issues
        pl_module.eval()

        # Determine output directory and filename format
        if self.separate_epoch_dirs:
            epoch_output_dir = os.path.join(self.output_dir, stage, f"epoch_{trainer.current_epoch+1}")
            epoch_suffix = ""  # No suffix needed when using separate directories
        else:
            epoch_output_dir = os.path.join(self.output_dir, stage)
            epoch_suffix = f"_epoch{trainer.current_epoch+1}" if self.include_epoch_in_filename else ""

        os.makedirs(epoch_output_dir, exist_ok=True)

        saved_ligand_count = 0
        saved_protein_count = 0

        # Use torch.no_grad() instead of inference_mode to avoid the tensor update error
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

                # Move batch to correct device - be careful with device placement
                batch_on_device = self._move_batch_to_device(batch, pl_module.device)

                try:
                    # Get predictions - make sure tensors are detached properly
                    results = pl_module(batch_on_device)

                    # print(f"SAVER DEBUG: Model output keys: {list(results.keys())}")

                    # Check the structure of results
                    # if 'ligand_coords' in results:
                    #     print(f"SAVER DEBUG: ligand_coords keys: {list(results['ligand_coords'].keys())}")
                    if 'sidechain_predictions' in results:
                        # print(f"SAVER DEBUG: sidechain_predictions keys: {list(results['sidechain_predictions'].keys())}")
                        # Show structure of first batch
                        first_batch_key = list(results['sidechain_predictions'].keys())[0]
                        first_residues = list(results['sidechain_predictions'][first_batch_key].keys())[:3]
                        # print(f"SAVER DEBUG: First batch residues: {first_residues}")

                    # Process each item in the batch
                    batch_size = self._determine_batch_size(results, batch)
                    # print(f"SAVER DEBUG: Determined batch size: {batch_size}")

                    for b_idx in range(batch_size):
                        complex_name = batch['name'][b_idx]
                        # print(f"SAVER DEBUG: Processing complex {complex_name} (batch index {b_idx})")

                        # Save ligand coordinates (new format)
                        if 'ligand_coords' in results and results['ligand_coords']:
                            if b_idx in results['ligand_coords']:
                                ligand_data = results['ligand_coords'][b_idx]
                                if isinstance(ligand_data, dict) and 'predictions' in ligand_data:
                                    pred_coords = ligand_data['predictions'].detach().cpu()
                                    # print(f"SAVER DEBUG: Saving ligand coords for {complex_name}, shape: {pred_coords.shape}")
                                    success = save_predicted_mol2_sidechain_map(
                                        name=complex_name,
                                        predicted_coords=pred_coords,
                                        original_mol2_dir=self.original_mol2_dir,
                                        output_mol2_dir=epoch_output_dir,
                                        epoch_suffix=epoch_suffix
                                    )
                                    if success:
                                        saved_ligand_count += 1
                                    # else:
                                    #     print(f"SAVER DEBUG: Failed to save ligand for {complex_name}")
                            #     else:
                            #         print(f"SAVER DEBUG: Invalid ligand data structure for {complex_name}")
                            # else:
                            #     print(f"SAVER DEBUG: No ligand coords for batch index {b_idx}")

                        # Save protein sidechain coordinates (new format)
                        if 'sidechain_predictions' in results and results['sidechain_predictions']:
                            if b_idx in results['sidechain_predictions']:
                                sidechain_predictions = results['sidechain_predictions'][b_idx]
                                sidechain_targets = results['sidechain_targets'][b_idx] if 'sidechain_targets' in results and b_idx in results['sidechain_targets'] else {}

                                # print(f"SAVER DEBUG: Saving sidechain coords for {complex_name}")
                                # print(f"SAVER DEBUG: Predictions residues: {len(sidechain_predictions)}")
                                # print(f"SAVER DEBUG: Targets residues: {len(sidechain_targets)}")

                                success = save_predicted_pdb_sidechain_map(
                                    name=complex_name,
                                    sidechain_predictions=sidechain_predictions,
                                    sidechain_targets=sidechain_targets,
                                    original_pdb_dir=self.original_pdb_dir,
                                    output_pdb_dir=epoch_output_dir,
                                    epoch_suffix=epoch_suffix
                                )
                                if success:
                                    saved_protein_count += 1
                            #     else:
                            #         print(f"SAVER DEBUG: Failed to save protein for {complex_name}")
                            # else:
                            #     print(f"SAVER DEBUG: No sidechain predictions for batch index {b_idx}")

                except Exception as e:
                    print(f"SAVER ERROR: Error processing batch {batch_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        print(f"Epoch {trainer.current_epoch + 1} Summary: Saved {saved_ligand_count} ligand structures and {saved_protein_count} protein structures.")
        pl_module.train()

    def _move_batch_to_device(self, batch, device):
        """Safely move batch to device"""
        batch_on_device = {}
        for key, value in batch.items():
            if key in ['protein_only', 'protein_virtual', 'ligand', 'affinity']:
                if hasattr(value, 'to'):
                    batch_on_device[key] = value.to(device)
                else:
                    batch_on_device[key] = value
            else:
                # Keep sidechain_map, name, etc. as-is
                batch_on_device[key] = value
        return batch_on_device

    def _determine_batch_size(self, results, batch):
        """Determine batch size from available data"""
        # First check the batch names
        if 'name' in batch and isinstance(batch['name'], (list, tuple)):
            return len(batch['name'])

        # Then check results
        if 'ligand_coords' in results and results['ligand_coords']:
            return len(results['ligand_coords'])
        elif 'sidechain_predictions' in results and results['sidechain_predictions']:
            return len(results['sidechain_predictions'])
        else:
            # Fallback
            return 1


# --- Updated Saving Functions for Sidechain Map Format ---

def save_predicted_mol2_sidechain_map(name, predicted_coords, original_mol2_dir, output_mol2_dir, epoch_suffix=""):
    """
    Save predicted ligand coordinates to MOL2 format.
    Compatible with the new sidechain_map EGNN output.
    """
    original_path = os.path.join(original_mol2_dir, f"{name}.qh.mol2")
    output_mol2_dir = os.path.join(output_mol2_dir, 'ligands')
    output_path = os.path.join(output_mol2_dir, f"{name}{epoch_suffix}.mol2")

    if not os.path.exists(original_path):
        print(f"Warning: Original MOL2 file not found: {original_path}")
        return False

    os.makedirs(output_mol2_dir, exist_ok=True)

    # Convert to numpy with proper device handling
    if torch.is_tensor(predicted_coords):
        coords_np = predicted_coords.detach().cpu().numpy()
    else:
        coords_np = np.array(predicted_coords)

    try:
        with open(original_path, 'r') as f:
            lines = f.readlines()

        # Find atom section
        atom_section_start = -1
        atom_section_end = -1
        for i, line in enumerate(lines):
            if line.strip() == "@<TRIPOS>ATOM":
                atom_section_start = i + 1
            elif line.strip().startswith("@<TRIPOS>") and atom_section_start != -1:
                atom_section_end = i
                break

        if atom_section_end == -1 and atom_section_start != -1:
            atom_section_end = len(lines)

        if atom_section_start == -1:
            print(f"Warning: No @<TRIPOS>ATOM section found in {original_path}")
            return False

        # Replace coordinates
        atom_count = 0
        modified_lines = list(lines)

        for i in range(atom_section_start, atom_section_end):
            if atom_count >= len(coords_np):
                break

            parts = modified_lines[i].split()
            if len(parts) >= 6:
                parts[2] = f"{coords_np[atom_count][0]:>9.4f}"
                parts[3] = f"{coords_np[atom_count][1]:>9.4f}"
                parts[4] = f"{coords_np[atom_count][2]:>9.4f}"
                modified_lines[i] = f"{parts[0]:>7s} {parts[1]:<8s} {parts[2]} {parts[3]} {parts[4]} {' '.join(parts[5:])}\n"
                atom_count += 1

        with open(output_path, 'w') as f:
            f.writelines(modified_lines)

        # print(f"Successfully saved ligand coordinates for {name} ({atom_count} atoms)")
        return True

    except Exception as e:
        print(f"Error saving MOL2 for {name}: {str(e)}")
        return False


def save_predicted_pdb_sidechain_map(name, sidechain_predictions, sidechain_targets,
                                   original_pdb_dir, output_pdb_dir, epoch_suffix=""):
    """
    Save predicted protein sidechain coordinates using sidechain_map format.

    Args:
        name: Complex name (e.g., "1a0o_A_1")
        sidechain_predictions: Dict {residue_key: {atom_name: tensor[3]}}
        sidechain_targets: Dict {residue_key: {atom_name: tensor[3]}} (optional)
        original_pdb_dir: Directory containing original PDB files
        output_pdb_dir: Directory to save predicted PDB files

    Returns:
        bool: True if successful, False otherwise
    """
    # Find original PDB file
    original_path = find_original_pdb_path(name, original_pdb_dir)
    if not original_path:
        return False

    output_pdb_dir = os.path.join(output_pdb_dir, 'proteins')
    os.makedirs(output_pdb_dir, exist_ok=True)
    output_path = os.path.join(output_pdb_dir, f"{name}{epoch_suffix}.pdb")

    try:
        # Convert tensor predictions to numpy with proper device handling
        residue_to_predicted_coords = {}

        # print(f"SAVER DEBUG: Processing {name}")
        # print(f"SAVER DEBUG: Sidechain predictions keys: {list(sidechain_predictions.keys())[:5]}...")  # Show first 5

        total_atoms = 0
        for residue_key, atom_dict in sidechain_predictions.items():
            #print(f"SAVER DEBUG: Processing residue {residue_key} with {len(atom_dict)} atoms")
            #SAVER DEBUG: Processing residue (ASN.A.280) with 4 atoms
            residue_coords = {}

            for atom_name, coord_tensor in atom_dict.items():
                if torch.is_tensor(coord_tensor):
                    coord_np = coord_tensor.detach().cpu().numpy()
                elif hasattr(coord_tensor, 'detach'):
                    coord_np = coord_tensor.detach().cpu().numpy()
                else:
                    coord_np = np.array(coord_tensor)
                residue_coords[atom_name] = coord_np
                total_atoms += 1

            if residue_coords:  # Only add if we have coordinates
                # Parse residue key: '(GLN.A.68)' -> 'A_68_GLN'
                res_name, chain_id, res_num = parse_residue_key(residue_key)
                if res_name and chain_id and res_num:
                    pdb_key = f"{chain_id}_{res_num}_{res_name}"
                    residue_to_predicted_coords[pdb_key] = residue_coords
        #         else:
        #             print(f"SAVER WARNING: Could not parse residue key: {residue_key}")

        # print(f"SAVER DEBUG: Total residues mapped: {len(residue_to_predicted_coords)}")
        # print(f"SAVER DEBUG: Total atoms: {total_atoms}")

        # Check for potential issues with predictions vs targets
        if sidechain_targets:
            identical_count = 0
            total_comparisons = 0
            for residue_key in sidechain_predictions:
                if residue_key in sidechain_targets:
                    for atom_name in sidechain_predictions[residue_key]:
                        if atom_name in sidechain_targets[residue_key]:
                            pred = sidechain_predictions[residue_key][atom_name]
                            target = sidechain_targets[residue_key][atom_name]

                            if torch.is_tensor(pred) and torch.is_tensor(target):
                                distance = torch.norm(pred - target).item()
                                total_comparisons += 1

                                # Check if prediction is identical to target (problematic!)
                                if distance < 0.001:
                                    identical_count += 1

            # if total_comparisons > 0:
            #     print(f"SAVER DEBUG: {identical_count}/{total_comparisons} predictions identical to targets ({100*identical_count/total_comparisons:.1f}%)")
            #     if identical_count == total_comparisons:
            #         print("SAVER WARNING: ALL predictions are identical to targets! Model may not be learning.")

        # Read original PDB file
        with open(original_path, 'r') as f:
            lines = f.readlines()

        # Process PDB lines and replace sidechain coordinates
        modified_lines = []
        atoms_replaced = 0

        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                modified_line, was_replaced = process_pdb_atom_line_sidechain_map(line, residue_to_predicted_coords)
                modified_lines.append(modified_line)
                if was_replaced:
                    atoms_replaced += 1
            else:
                # Keep non-ATOM lines as-is (headers, etc.)
                modified_lines.append(line)

        # print(f"SAVER DEBUG: Replaced coordinates for {atoms_replaced} atoms")

        # Write the modified PDB file
        with open(output_path, 'w') as f:
            f.writelines(modified_lines)

        # print(f"Successfully saved protein coordinates for {name}")
        return True

    except Exception as e:
        print(f"Error saving predicted PDB for {name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def process_pdb_atom_line_sidechain_map(line, residue_to_predicted_coords):
    """
    Process a single PDB ATOM/HETATM line, replacing sidechain coordinates if available.
    Returns (modified_line, was_replaced)
    """
    # Parse PDB line
    atom_name = line[12:16].strip()
    res_name = line[17:20].strip()
    chain_id = line[21:22].strip()
    res_num_str = line[22:26].strip()

    try:
        # Handle insertion codes - extract just the numeric part
        res_num = ''.join(c for c in res_num_str if c.isdigit() or c == '-')
        if not res_num:
            return line, False  # Keep original if can't parse
    except:
        return line, False

    pdb_key = f"{chain_id}_{res_num}_{res_name}"

    # Check if this is a sidechain atom we want to replace
    if (pdb_key in residue_to_predicted_coords and
        atom_name in residue_to_predicted_coords[pdb_key] and
        atom_name not in ['N', 'CA', 'C', 'O']):  # Preserve backbone atoms

        # Replace coordinates with predicted ones
        new_coords = residue_to_predicted_coords[pdb_key][atom_name]
        new_line = (
            line[:30] +  # Keep everything up to coordinates
            f"{new_coords[0]:8.3f}{new_coords[1]:8.3f}{new_coords[2]:8.3f}" +
            line[54:]   # Keep everything after coordinates
        )
        return new_line, True
    else:
        # Keep original coordinates for backbone atoms or unmatched atoms
        return line, False


def find_original_pdb_path(name, original_pdb_dir):
    """Find the original PDB file using various naming conventions."""
    # Try direct naming first
    direct_path = os.path.join(original_pdb_dir, f"{name}.pdb")
    if os.path.exists(direct_path):
        return direct_path

    # Try BioLiP naming convention: "5e4t_MBT_A_1" -> "5e4tA.pdb"
    name_parts = name.split('_')
    if len(name_parts) >= 3:
        pdb_code = name_parts[0]
        chain_id = name_parts[2]
        biolip_name = f"{pdb_code}{chain_id}.pdb"
        biolip_path = os.path.join(original_pdb_dir, biolip_name)

        if os.path.exists(biolip_path):
            return biolip_path

        # Try other alternative naming conventions
        alt_paths = [
            os.path.join(original_pdb_dir, f"{name}_receptor.pdb"),
            os.path.join(original_pdb_dir, f"{name}_protein.pdb"),
            os.path.join(original_pdb_dir, f"{pdb_code}.pdb"),
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                return alt_path

    print(f"Warning: Original PDB file not found for {name}")
    return None

    
def parse_residue_key(residue_key):
    """Parse residue key from sidechain_map format: '(GLN.A.68)' -> ('GLN', 'A', '68')"""
    residue_key = str(residue_key)
    try:
        clean_key = residue_key.strip('()')
        parts = clean_key.split('.')
        if len(parts) >= 3:
            res_name = parts[0]
            chain_id = parts[1]
            res_num = parts[2]
            return res_name, chain_id, res_num
    except:
        pass
    return None, None, None


# --- Additional Utility Functions ---

def save_coordinate_tensors(results, epoch_output_dir, stage='val'):
    """
    Save raw coordinate tensors as .pt files for debugging.
    """
    tensor_dir = os.path.join(epoch_output_dir, 'tensors')
    os.makedirs(tensor_dir, exist_ok=True)

    # Save ligand coordinates
    if 'ligand_coords' in results and results['ligand_coords']:
        ligand_tensor_data = {}
        for batch_id, batch_data in results['ligand_coords'].items():
            if isinstance(batch_data, dict):
                # Convert tensors to CPU before saving
                ligand_tensor_data[batch_id] = {
                    'predictions': batch_data['predictions'].detach().cpu() if torch.is_tensor(batch_data['predictions']) else batch_data['predictions'],
                    'targets': batch_data['targets'].detach().cpu() if torch.is_tensor(batch_data['targets']) else batch_data['targets']
                }
        torch.save(ligand_tensor_data, os.path.join(tensor_dir, f'{stage}_ligand_coords.pt'))

    # Save sidechain coordinates
    if 'sidechain_predictions' in results and results['sidechain_predictions']:
        sidechain_tensor_data = {}
        for batch_id, batch_predictions in results['sidechain_predictions'].items():
            batch_data_cpu = {}
            for residue_key, atom_dict in batch_predictions.items():
                residue_data_cpu = {}
                for atom_name, coord_tensor in atom_dict.items():
                    if torch.is_tensor(coord_tensor):
                        residue_data_cpu[atom_name] = coord_tensor.detach().cpu()
                    else:
                        residue_data_cpu[atom_name] = coord_tensor
                batch_data_cpu[residue_key] = residue_data_cpu
            sidechain_tensor_data[batch_id] = batch_data_cpu
        torch.save(sidechain_tensor_data, os.path.join(tensor_dir, f'{stage}_sidechain_coords.pt'))


# --- Early Stopping with Delayed Start ---

class DelayedEarlyStopping(EarlyStopping):
    """EarlyStopping callback that only starts monitoring after a certain number of epochs."""
    def __init__(self, min_epochs_before_stop=10, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs_before_stop = min_epochs_before_stop

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.min_epochs_before_stop:
            return
        super().on_validation_end(trainer, pl_module)
