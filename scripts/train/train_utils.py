# MOL2 coordinate replacement functions

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from biopandas.pdb import PandasPdb

class CoordinateSaverCallback(Callback):
    """
    A PyTorch Lightning Callback to save predicted coordinates for both ligands (MOL2)
    and protein sidechains (PDB) during training and testing.
    Uses sidechain_map for accurate residue-atom mapping.
    """
    def __init__(self,
                 save_every_n_epochs=20,
                 original_mol2_dir="/home/j2ho/DB/biolip/BioLiP_updated_set/ligand_mol2",
                 original_pdb_dir="/home/j2ho/DB/biolip/BioLiP_updated_set/receptor",
                 output_dir="./predictions",
                 save_coords_pt=True,
                 separate_epoch_dirs=True):
        super().__init__()
        self.save_every_n_epochs = save_every_n_epochs
        self.original_mol2_dir = original_mol2_dir
        self.original_pdb_dir = original_pdb_dir
        self.output_dir = output_dir
        self.save_coords_pt = save_coords_pt
        self.separate_epoch_dirs = separate_epoch_dirs

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
        Main method to orchestrate saving coordinates for a given stage.
        """
        if stage == 'val':
            dataloader = trainer.val_dataloaders
        elif stage == 'test':
            dataloader = trainer.test_dataloaders
        else:
            return

        pl_module.eval()
        
        # Determine the base output directory for this stage and epoch
        if self.separate_epoch_dirs:
            epoch_output_dir = os.path.join(self.output_dir, stage, f"epoch_{trainer.current_epoch+1}")
        else:
            epoch_output_dir = os.path.join(self.output_dir, stage)
        os.makedirs(epoch_output_dir, exist_ok=True)
        
        saved_ligand_count = 0
        saved_protein_count = 0

        with torch.inference_mode(False): 
            for batch in dataloader:
                # Move batch to the correct device
                batch_on_device = {
                    'protein_only': batch['protein_only'].to(pl_module.device),
                    'protein_virtual': batch['protein_virtual'].to(pl_module.device),
                    'ligand': batch['ligand'].to(pl_module.device),
                    'affinity': batch['affinity'].to(pl_module.device),
                    'name': batch['name'],
                    'sidechain_map': batch['sidechain_map']  # Keep sidechain_map
                }

                results = pl_module(batch_on_device)
                batch_size = results['affinity'].size(0)

                for b_idx in range(batch_size):
                    complex_name = batch['name'][b_idx]

                    # --- 1. Save Ligand Coordinates ---
                    if 'predicted_ligand_coords' in results and results['predicted_ligand_coords'] is not None:
                        lig_mask = results['lig_coord_target_mask'][b_idx]
                        num_lig_atoms = lig_mask.sum().item()
                        if num_lig_atoms > 0:
                            pred_lig_coords = results['predicted_ligand_coords'][b_idx][:num_lig_atoms]
                            success = save_predicted_mol2(
                                name=complex_name,
                                predicted_coords=pred_lig_coords,
                                original_mol2_dir=self.original_mol2_dir,
                                output_mol2_dir=epoch_output_dir
                            )
                            if success: saved_ligand_count += 1
                    
                    # --- 2. Save Protein Sidechain Coordinates ---
                    if 'predicted_sidechain_coords' in results and results['predicted_sidechain_coords'] is not None:
                        # Get sidechain_map for this batch
                        if isinstance(batch['sidechain_map'], (tuple, list)):
                            batch_sidechain_map = batch['sidechain_map'][b_idx]
                        else:
                            batch_sidechain_map = batch['sidechain_map']
                        
                        success = save_predicted_pdb_with_sidechain_map(
                            name=complex_name,
                            predicted_sidechain_coords=results['predicted_sidechain_coords'][b_idx],
                            prot_res_mask=results['prot_coord_target_mask'][b_idx],
                            sidechain_atom_mask=results['sidechain_mask'][b_idx],
                            sidechain_map=batch_sidechain_map,
                            original_pdb_dir=self.original_pdb_dir,
                            output_pdb_dir=epoch_output_dir
                        )
                        if success: saved_protein_count += 1

        print(f"Epoch {trainer.current_epoch + 1} Summary: Saved {saved_ligand_count} ligand structures and {saved_protein_count} protein structures.")
        pl_module.train()


# --- Standalone File Saving Functions ---

def save_predicted_mol2(name, predicted_coords, original_mol2_dir, output_mol2_dir):
    """
    Replace coordinates in an original MOL2 file with predicted coordinates.
    """
    original_path = os.path.join(original_mol2_dir, f"{name}.qh.mol2")
    output_mol2_dir = os.path.join(output_mol2_dir, 'ligands')
    output_path = os.path.join(output_mol2_dir, f"{name}.mol2")
    if not os.path.exists(original_path):
        return False
    os.makedirs(output_mol2_dir, exist_ok=True)
        
    coords_np = predicted_coords.detach().numpy()

    with open(original_path, 'r') as f:
        lines = f.readlines()
        
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
        return False

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
    
    return True


def save_predicted_pdb_with_sidechain_map(name, predicted_sidechain_coords, prot_res_mask, 
                                         sidechain_atom_mask, sidechain_map, 
                                         original_pdb_dir, output_pdb_dir):
    """
    Save predicted protein sidechain coordinates to a PDB file using sidechain_map
    for accurate residue-atom mapping.
    
    Args:
        name: Complex name (e.g., "1a0o_A_1")
        predicted_sidechain_coords: Tensor of shape [max_residues, max_sidechain_atoms, 3]
        prot_res_mask: Boolean mask of shape [max_residues] indicating valid residues
        sidechain_atom_mask: Boolean mask of shape [max_residues, max_sidechain_atoms] indicating valid atoms
        sidechain_map: Dict mapping residue keys to atom coordinates {'(GLN.A.68)': {'CB': coords, ...}}
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
    output_path = os.path.join(output_pdb_dir, f"{name}.pdb")
    
    try:
        # Convert tensors to numpy
        pred_coords_np = predicted_sidechain_coords.detach().numpy()
        prot_mask_np = prot_res_mask.detach().numpy()
        sidechain_mask_np = sidechain_atom_mask.detach().numpy()
        
        # Read original PDB file
        with open(original_path, 'r') as f:
            lines = f.readlines()
        
        # Create mapping from predicted coordinates to sidechain_map structure
        residue_to_predicted_coords = create_residue_coord_mapping(
            sidechain_map, pred_coords_np, prot_mask_np, sidechain_mask_np
        )
        
        # Process PDB lines and replace sidechain coordinates
        modified_lines = []
        
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                modified_line = process_pdb_atom_line(line, residue_to_predicted_coords)
                modified_lines.append(modified_line)
            else:
                # Keep non-ATOM lines as-is (headers, etc.)
                modified_lines.append(line)
        
        # Write the modified PDB file
        with open(output_path, 'w') as f:
            f.writelines(modified_lines)
        
        return True
        
    except Exception as e:
        print(f"Error saving predicted PDB for {name}: {str(e)}")
        return False


def find_original_pdb_path(name, original_pdb_dir):
    """
    Find the original PDB file using various naming conventions.
    
    Args:
        name: Complex name (e.g., "5e4t_MBT_A_1")
        original_pdb_dir: Directory containing original PDB files
    
    Returns:
        str or None: Path to original PDB file if found, None otherwise
    """
    # Try direct naming first
    direct_path = os.path.join(original_pdb_dir, f"{name}.pdb")
    if os.path.exists(direct_path):
        return direct_path
    
    # Try BioLiP naming convention: "5e4t_MBT_A_1" -> "5e4tA.pdb"
    name_parts = name.split('_')
    if len(name_parts) >= 3:
        pdb_code = name_parts[0]  # "5e4t"
        chain_id = name_parts[2]  # "A"
        biolip_name = f"{pdb_code}{chain_id}.pdb"  # "5e4tA.pdb"
        biolip_path = os.path.join(original_pdb_dir, biolip_name)
        
        if os.path.exists(biolip_path):
            return biolip_path
        
        # Try other alternative naming conventions
        alt_paths = [
            os.path.join(original_pdb_dir, f"{name}_receptor.pdb"),
            os.path.join(original_pdb_dir, f"{name}_protein.pdb"),
            os.path.join(original_pdb_dir, f"{pdb_code}.pdb"),  # Just PDB code
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                return alt_path
    
    print(f"Warning: Original PDB file not found for {name}")
    return None


def create_residue_coord_mapping(sidechain_map, pred_coords_np, prot_mask_np, sidechain_mask_np):
    """
    Create mapping from PDB residue identifiers to predicted coordinates.
    
    Args:
        sidechain_map: Dict with residue keys like '(GLN.A.68)' and atom coordinate values
        pred_coords_np: Predicted coordinates array [max_residues, max_sidechain_atoms, 3]
        prot_mask_np: Boolean mask indicating valid residues
        sidechain_mask_np: Boolean mask indicating valid atoms
    
    Returns:
        dict: Mapping from 'CHAIN_RESNUM_RESNAME' to atom coordinate dict
    """
    residue_to_coords = {}
    residue_keys = list(sidechain_map.keys())
    
    valid_residue_idx = 0
    for res_idx, is_valid_res in enumerate(prot_mask_np):
        if not is_valid_res or res_idx >= len(residue_keys):
            continue
            
        if valid_residue_idx >= pred_coords_np.shape[0]:
            break
        
        # Parse residue key: '(GLN.A.68)' -> 'A_68_GLN'
        residue_key = residue_keys[res_idx]
        res_name, chain_id, res_num = parse_residue_key(residue_key)
        
        if res_name and chain_id and res_num:
            pdb_key = f"{chain_id}_{res_num}_{res_name}"
            
            # Get atoms from sidechain_map for this residue
            original_atoms = sidechain_map[residue_key]
            
            # Map predicted coordinates to atom names
            predicted_coords_for_res = pred_coords_np[valid_residue_idx]
            atom_mask_for_res = sidechain_mask_np[valid_residue_idx]
            
            atom_coords = {}
            atom_names = list(original_atoms.keys())
            
            for atom_idx, atom_name in enumerate(atom_names):
                if (atom_idx < len(atom_mask_for_res) and 
                    atom_mask_for_res[atom_idx] and 
                    atom_idx < predicted_coords_for_res.shape[0]):
                    atom_coords[atom_name] = predicted_coords_for_res[atom_idx]
            
            if atom_coords:  # Only add if we have predicted coordinates
                residue_to_coords[pdb_key] = atom_coords
        
        valid_residue_idx += 1
    
    return residue_to_coords


def parse_residue_key(residue_key):
    """
    Parse residue key from sidechain_map format.
    
    Args:
        residue_key: String like '(GLN.A.68)'
    
    Returns:
        tuple: (res_name, chain_id, res_num) or (None, None, None) if parsing fails
    """
    try:
        # Remove parentheses and split by dots
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


def process_pdb_atom_line(line, residue_to_predicted_coords):
    """
    Process a single PDB ATOM/HETATM line, replacing sidechain coordinates if available.
    
    Args:
        line: PDB line string
        residue_to_predicted_coords: Dict mapping residue keys to atom coordinates
    
    Returns:
        str: Modified PDB line
    """
    # Parse PDB line
    atom_name = line[12:16].strip()
    res_name = line[17:20].strip()
    chain_id = line[21:22].strip()
    res_num_str = line[22:26].strip()
    
    try:
        # Handle insertion codes
        res_num = ''.join(c for c in res_num_str if c.isdigit() or c == '-')
        if not res_num:
            return line  # Keep original if can't parse
    except:
        return line
    
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
        return new_line
    else:
        # Keep original coordinates for backbone atoms or unmatched atoms
        return line
# --- Constants and Helper Dictionaries ---

# This dictionary is critical for mapping predicted coordinates to the correct atoms.
# The order of atom names for each residue MUST match the order used to generate
# the 'X_sidechain_padded' tensor in dataset preparation.

SIDECHAIN_ATOMS = {
    'ALA': ['CB'],
    'ARG': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'ASN': ['CB', 'CG', 'OD1', 'ND2'],
    'ASP': ['CB', 'CG', 'OD1', 'OD2'],
    'CYS': ['CB', 'SG'],
    'GLN': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
    'GLU': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
    'HIS': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'ILE': ['CB', 'CG1', 'CG2', 'CD1'],
    'LEU': ['CB', 'CG', 'CD1', 'CD2'],
    'LYS': ['CB', 'CG', 'CD', 'CE', 'NZ'],
    'MET': ['CB', 'CG', 'SD', 'CE'],
    'PHE': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'PRO': ['CB', 'CG', 'CD'],
    'SER': ['CB', 'OG'],
    'THR': ['CB', 'OG1', 'CG2'],
    'TRP': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'VAL': ['CB', 'CG1', 'CG2'],
    'GLY': [] # Glycine has no sidechain atoms beyond C-alpha
}

# --- Pytorch Lightning Callbacks ---

class DelayedEarlyStopping(EarlyStopping):
    """
    An EarlyStopping callback that only starts monitoring after a certain number of epochs.
    """
    def __init__(self, min_epochs_before_stop=10, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs_before_stop = min_epochs_before_stop

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.min_epochs_before_stop:
            return
        super().on_validation_end(trainer, pl_module)
'''
class CoordinateSaverCallback(Callback):
    """
    A PyTorch Lightning Callback to save predicted coordinates for both ligands (MOL2)
    and protein sidechains (PDB) during training and testing.
    """
    def __init__(self,
                 save_every_n_epochs=20,
                 original_mol2_dir="/home/j2ho/DB/biolip/BioLiP_updated_set/ligand_mol2",
                 original_pdb_dir="/home/j2ho/DB/biolip/BioLiP_updated_set/receptor", # Corrected path
                 output_dir="./predictions",
                 save_coords_pt=True,
                 separate_epoch_dirs=True):
        super().__init__()
        self.save_every_n_epochs = save_every_n_epochs
        self.original_mol2_dir = original_mol2_dir
        self.original_pdb_dir = original_pdb_dir
        self.output_dir = output_dir
        self.save_coords_pt = save_coords_pt
        self.separate_epoch_dirs = separate_epoch_dirs

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
        Main method to orchestrate saving coordinates for a given stage.
        """
        if stage == 'val':
            dataloader = trainer.val_dataloaders
        elif stage == 'test':
            dataloader = trainer.test_dataloaders
        else:
            return

        pl_module.eval()
        
        # Determine the base output directory for this stage and epoch
        if self.separate_epoch_dirs:
            epoch_output_dir = os.path.join(self.output_dir, stage, f"epoch_{trainer.current_epoch+1}")
        else:
            epoch_output_dir = os.path.join(self.output_dir, stage)
        os.makedirs(epoch_output_dir, exist_ok=True)
        
        saved_ligand_count = 0
        saved_protein_count = 0

        # with torch.no_grad():
        with torch.inference_mode(False): 
            for batch in dataloader:
                # Move batch to the correct device
                batch_on_device = {
                    'protein_only': batch['protein_only'].to(pl_module.device),
                    'protein_virtual': batch['protein_virtual'].to(pl_module.device),
                    'ligand': batch['ligand'].to(pl_module.device),
                    'affinity': batch['affinity'].to(pl_module.device),
                    'name': batch['name']
                }

                results = pl_module(batch_on_device)
                batch_size = results['affinity'].size(0)

                for b_idx in range(batch_size):
                    complex_name = batch['name'][b_idx]

                    # --- 1. Save Ligand Coordinates ---
                    if 'predicted_ligand_coords' in results and results['predicted_ligand_coords'] is not None:
                        lig_mask = results['lig_coord_target_mask'][b_idx]
                        num_lig_atoms = lig_mask.sum().item()
                        if num_lig_atoms > 0:
                            pred_lig_coords = results['predicted_ligand_coords'][b_idx][:num_lig_atoms]
                            success = save_predicted_mol2(
                                name=complex_name,
                                predicted_coords=pred_lig_coords,
                                original_mol2_dir=self.original_mol2_dir,
                                output_mol2_dir=epoch_output_dir
                            )
                            if success: saved_ligand_count += 1
                    
                    # --- 2. Save Protein Sidechain Coordinates ---
                    if 'predicted_sidechain_coords' in results and results['predicted_sidechain_coords'] is not None:
                        res_list_single = batch['protein_virtual'].res_list[b_idx]
                        
                        success = save_predicted_pdb(
                            name=complex_name,
                            predicted_sidechain_coords=results['predicted_sidechain_coords'][b_idx],
                            prot_res_mask=results['prot_coord_target_mask'][b_idx],
                            sidechain_atom_mask=results['sidechain_mask'][b_idx],
                            res_list=res_list_single,
                            original_pdb_dir=self.original_pdb_dir,
                            output_pdb_dir=epoch_output_dir
                        )
                        if success: saved_protein_count += 1

        print(f"Epoch {trainer.current_epoch + 1} Summary: Saved {saved_ligand_count} ligand structures and {saved_protein_count} protein structures.")
        pl_module.train()

# --- Standalone File Saving Functions ---

def save_predicted_mol2(name, predicted_coords, original_mol2_dir, output_mol2_dir):
    """
    Replace coordinates in an original MOL2 file with predicted coordinates.
    """
    original_path = os.path.join(original_mol2_dir, f"{name}.qh.mol2")
    output_mol2_dir = os.path.join(output_mol2_dir, 'ligands')
    output_path = os.path.join(output_mol2_dir, f"{name}.mol2")
    if not os.path.exists(original_path):
        return False
    # create output directory if it doesn't exist
    os.makedirs(output_mol2_dir, exist_ok=True)
        
    coords_np = predicted_coords.cpu().numpy()

    with open(original_path, 'r') as f:
        lines = f.readlines()
        
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
        return False

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
    
    return True

# Add this function to your existing train_utils.py file
# Place it after the save_predicted_mol2 function

def save_predicted_pdb(name, predicted_sidechain_coords, prot_res_mask, 
                       sidechain_atom_mask, res_list, original_pdb_dir, output_pdb_dir):
    """
    Save predicted protein sidechain coordinates to a PDB file while preserving backbone coordinates.
    
    Args:
        name: Complex name (e.g., "1a0o_A_1")
        predicted_sidechain_coords: Tensor of shape [max_residues, max_sidechain_atoms, 3]
        prot_res_mask: Boolean mask of shape [max_residues] indicating valid residues
        sidechain_atom_mask: Boolean mask of shape [max_residues, max_sidechain_atoms] indicating valid atoms
        res_list: List of Residue objects from the original structure
        original_pdb_dir: Directory containing original PDB files
        output_pdb_dir: Directory to save predicted PDB files
    
    Returns:
        bool: True if successful, False otherwise
    """
    import numpy as np
    
    # Find original PDB file
    # Handle naming convention: "5e4t_MBT_A_1" -> "5e4tA.pdb"
    original_path = os.path.join(original_pdb_dir, f"{name}.pdb")
    
    if not os.path.exists(original_path):
        # Try BioLiP naming convention: extract PDB code and chain
        # Format: "5e4t_MBT_A_1" -> "5e4tA.pdb"
        name_parts = name.split('_')
        if len(name_parts) >= 3:
            pdb_code = name_parts[0]  # "5e4t"
            chain_id = name_parts[2]  # "A"
            biolip_name = f"{pdb_code}{chain_id}.pdb"  # "5e4tA.pdb"
            biolip_path = os.path.join(original_pdb_dir, biolip_name)
            
            if os.path.exists(biolip_path):
                original_path = biolip_path
            else:
                # Try other alternative naming conventions
                alt_paths = [
                    os.path.join(original_pdb_dir, f"{name}_receptor.pdb"),
                    os.path.join(original_pdb_dir, f"{name}_protein.pdb"),
                    os.path.join(original_pdb_dir, f"{pdb_code}.pdb"),  # Just PDB code
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        original_path = alt_path
                        break
                else:
                    print(f"Warning: Original PDB file not found for {name}")
                    print(f"Tried paths: {name}.pdb, {biolip_name}, and alternatives")
                    return False
        else:
            print(f"Warning: Cannot parse name format '{name}' for PDB file lookup")
            return False
    
    output_pdb_dir = os.path.join(output_pdb_dir, 'proteins')
    os.makedirs(output_pdb_dir, exist_ok=True)  # Ensure output directory exists
    output_path = os.path.join(output_pdb_dir, f"{name}.pdb")
    
    try:
        # Convert tensors to numpy
        pred_coords_np = predicted_sidechain_coords.cpu().numpy()
        prot_mask_np = prot_res_mask.cpu().numpy()
        sidechain_mask_np = sidechain_atom_mask.cpu().numpy()
        
        # Read original PDB file
        with open(original_path, 'r') as f:
            lines = f.readlines()
        
        # Create a mapping from residue identifiers to predicted coordinates
        residue_to_coords = {}
        valid_residue_count = 0
        
        for res_idx, (is_valid_res, residue) in enumerate(zip(prot_mask_np, res_list)):
            if not is_valid_res or res_idx >= len(res_list):
                continue
                
            if valid_residue_count >= pred_coords_np.shape[0]:
                break
                
            # Get the residue identifier
            res_key = f"{residue.chain_id}_{residue.res_num}_{residue.res_name}"
            
            # Get sidechain atoms for this residue type
            if residue.res_name in SIDECHAIN_ATOMS:
                sidechain_atom_names = SIDECHAIN_ATOMS[residue.res_name]
                predicted_coords_for_res = pred_coords_np[valid_residue_count]
                atom_mask_for_res = sidechain_mask_np[valid_residue_count]
                
                # Map atom names to coordinates
                atom_coords = {}
                for atom_idx, atom_name in enumerate(sidechain_atom_names):
                    if atom_idx < len(atom_mask_for_res) and atom_mask_for_res[atom_idx]:
                        if atom_idx < predicted_coords_for_res.shape[0]:
                            atom_coords[atom_name] = predicted_coords_for_res[atom_idx]
                
                residue_to_coords[res_key] = atom_coords
            
            valid_residue_count += 1
        
        # Process PDB lines
        modified_lines = []
        
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                # Parse PDB line
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21:22].strip()
                res_num_str = line[22:26].strip()
                
                try:
                    # Handle insertion codes
                    res_num = int(''.join(c for c in res_num_str if c.isdigit() or c == '-'))
                except ValueError:
                    # Keep original line if parsing fails
                    modified_lines.append(line)
                    continue
                
                res_key = f"{chain_id}_{res_num}_{res_name}"
                
                # Check if this is a sidechain atom we want to replace
                if (res_key in residue_to_coords and 
                    atom_name in residue_to_coords[res_key] and
                    atom_name not in ['N', 'CA', 'C', 'O']):  # Preserve backbone atoms
                    
                    # Replace coordinates with predicted ones
                    new_coords = residue_to_coords[res_key][atom_name]
                    new_line = (
                        line[:30] +  # Keep everything up to coordinates
                        f"{new_coords[0]:8.3f}{new_coords[1]:8.3f}{new_coords[2]:8.3f}" +
                        line[54:]   # Keep everything after coordinates
                    )
                    modified_lines.append(new_line)
                else:
                    # Keep original coordinates for backbone atoms or unmatched atoms
                    modified_lines.append(line)
            else:
                # Keep non-ATOM lines as-is (headers, etc.)
                modified_lines.append(line)
        
        # Write the modified PDB file
        with open(output_path, 'w') as f:
            f.writelines(modified_lines)
        
        return True
        
    except Exception as e:
        print(f"Error saving predicted PDB for {name}: {str(e)}")
        return False
    
def save_predicted_pdb_use_map(name, predicted_sidechain_coords, prot_res_mask, 
                       sidechain_atom_mask, res_list, original_pdb_dir, output_pdb_dir):
    """
    Save predicted protein sidechain coordinates to a PDB file while preserving backbone coordinates.
    
    Args:
        name: Complex name (e.g., "1a0o_A_1")
        predicted_sidechain_coords: Tensor of shape [max_residues, max_sidechain_atoms, 3]
        prot_res_mask: Boolean mask of shape [max_residues] indicating valid residues
        sidechain_atom_mask: Boolean mask of shape [max_residues, max_sidechain_atoms] indicating valid atoms
        res_list: List of Residue objects from the original structure
        original_pdb_dir: Directory containing original PDB files
        output_pdb_dir: Directory to save predicted PDB files
    
    Returns:
        bool: True if successful, False otherwise
    """
    import os
    import numpy as np
    
    # Find original PDB file
    # Handle naming convention: "5e4t_MBT_A_1" -> "5e4tA.pdb"
    original_path = os.path.join(original_pdb_dir, f"{name}.pdb")
    
    if not os.path.exists(original_path):
        # Try BioLiP naming convention: extract PDB code and chain
        # Format: "5e4t_MBT_A_1" -> "5e4tA.pdb"
        name_parts = name.split('_')
        if len(name_parts) >= 3:
            pdb_code = name_parts[0]  # "5e4t"
            chain_id = name_parts[2]  # "A"
            biolip_name = f"{pdb_code}{chain_id}.pdb"  # "5e4tA.pdb"
            biolip_path = os.path.join(original_pdb_dir, biolip_name)
            
            if os.path.exists(biolip_path):
                original_path = biolip_path
            else:
                # Try other alternative naming conventions
                alt_paths = [
                    os.path.join(original_pdb_dir, f"{name}_receptor.pdb"),
                    os.path.join(original_pdb_dir, f"{name}_protein.pdb"),
                    os.path.join(original_pdb_dir, f"{pdb_code}.pdb"),  # Just PDB code
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        original_path = alt_path
                        break
                else:
                    print(f"Warning: Original PDB file not found for {name}")
                    print(f"Tried paths: {name}.pdb, {biolip_name}, and alternatives")
                    return False
        else:
            print(f"Warning: Cannot parse name format '{name}' for PDB file lookup")
            return False
    
    output_path = os.path.join(output_pdb_dir, 'pdb', f"{name}_pred_protein.pdb")
    
    try:
        # Convert tensors to numpy
        pred_coords_np = predicted_sidechain_coords.cpu().numpy()
        prot_mask_np = prot_res_mask.cpu().numpy()
        sidechain_mask_np = sidechain_atom_mask.cpu().numpy()
        
        # Read original PDB file
        with open(original_path, 'r') as f:
            lines = f.readlines()
        
        # Create a mapping from residue identifiers to predicted coordinates
        residue_to_coords = {}
        valid_residue_count = 0
        
        for res_idx, (is_valid_res, residue) in enumerate(zip(prot_mask_np, res_list)):
            if not is_valid_res or res_idx >= len(res_list):
                continue
                
            if valid_residue_count >= pred_coords_np.shape[0]:
                break
                
            # Get the residue identifier
            res_key = f"{residue.chain_id}_{residue.res_num}_{residue.res_name}"
            
            # Get sidechain atoms for this residue type
            if residue.res_name in SIDECHAIN_ATOMS:
                sidechain_atom_names = SIDECHAIN_ATOMS[residue.res_name]
                predicted_coords_for_res = pred_coords_np[valid_residue_count]
                atom_mask_for_res = sidechain_mask_np[valid_residue_count]
                
                # Map atom names to coordinates
                atom_coords = {}
                for atom_idx, atom_name in enumerate(sidechain_atom_names):
                    if atom_idx < len(atom_mask_for_res) and atom_mask_for_res[atom_idx]:
                        if atom_idx < predicted_coords_for_res.shape[0]:
                            atom_coords[atom_name] = predicted_coords_for_res[atom_idx]
                
                residue_to_coords[res_key] = atom_coords
            
            valid_residue_count += 1
        
        # Process PDB lines
        modified_lines = []
        
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                # Parse PDB line
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21:22].strip()
                res_num_str = line[22:26].strip()
                
                try:
                    # Handle insertion codes
                    res_num = int(''.join(c for c in res_num_str if c.isdigit() or c == '-'))
                except ValueError:
                    # Keep original line if parsing fails
                    modified_lines.append(line)
                    continue
                
                res_key = f"{chain_id}_{res_num}_{res_name}"
                
                # Check if this is a sidechain atom we want to replace
                if (res_key in residue_to_coords and 
                    atom_name in residue_to_coords[res_key] and
                    atom_name not in ['N', 'CA', 'C', 'O']):  # Preserve backbone atoms
                    
                    # Replace coordinates with predicted ones
                    new_coords = residue_to_coords[res_key][atom_name]
                    new_line = (
                        line[:30] +  # Keep everything up to coordinates
                        f"{new_coords[0]:8.3f}{new_coords[1]:8.3f}{new_coords[2]:8.3f}" +
                        line[54:]   # Keep everything after coordinates
                    )
                    modified_lines.append(new_line)
                else:
                    # Keep original coordinates for backbone atoms or unmatched atoms
                    modified_lines.append(line)
            else:
                # Keep non-ATOM lines as-is (headers, etc.)
                modified_lines.append(line)
        
        # Write the modified PDB file
        with open(output_path, 'w') as f:
            f.writelines(modified_lines)
        
        return True
        
    except Exception as e:
        print(f"Error saving predicted PDB for {name}: {str(e)}")
        return False
'''