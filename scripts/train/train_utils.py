# MOL2 coordinate replacement functions

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, min_epochs_before_stop=10, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs_before_stop = min_epochs_before_stop

    def on_validation_end(self, trainer, pl_module):
        # Skip early stopping until min_epochs_before_stop
        if trainer.current_epoch < self.min_epochs_before_stop:
            return
        # After threshold, behave as normal
        super().on_validation_end(trainer, pl_module)
        
def save_predicted_mol2(name, predicted_coords, original_mol2_dir, output_mol2_dir):
    """
    Replace coordinates in original MOL2 file with predicted coordinates
    
    Args:
        name: Molecule name (e.g., "5e4t_MBT_A_1")
        predicted_coords: [num_atoms, 3] tensor of predicted coordinates
        original_mol2_dir: Directory containing original MOL2 files
        output_mol2_dir: Directory to save predicted MOL2 files
    """
    # Construct file paths
    original_path = os.path.join(original_mol2_dir, f"{name}.qh.mol2")
    output_path = os.path.join(output_mol2_dir, f"{name}.mol2")
    
    # Ensure output directory exists
    os.makedirs(output_mol2_dir, exist_ok=True)
    
    # Check if original file exists
    if not os.path.exists(original_path):
        print(f"Warning: Original MOL2 file not found: {original_path}")
        return False
    
    # Convert predicted coordinates to CPU numpy if needed
    if isinstance(predicted_coords, torch.Tensor):
        predicted_coords = predicted_coords.cpu().numpy()
    
    # Read original MOL2 file
    with open(original_path, 'r') as f:
        lines = f.readlines()
    
    # Find @<TRIPOS>ATOM section
    atom_section_start = None
    atom_section_end = None
    
    for i, line in enumerate(lines):
        if line.strip() == "@<TRIPOS>ATOM":
            atom_section_start = i + 1
        elif line.strip().startswith("@<TRIPOS>") and atom_section_start is not None:
            atom_section_end = i
            break
    
    if atom_section_start is None:
        print(f"Warning: Could not find @<TRIPOS>ATOM section in {original_path}")
        return False
    
    # If no end found, assume atoms go until end of file
    if atom_section_end is None:
        atom_section_end = len(lines)
    
    # Replace coordinates in atom lines
    atom_count = 0
    modified_lines = []
    
    for i, line in enumerate(lines):
        if atom_section_start <= i < atom_section_end:
            # This is an atom line - replace coordinates
            parts = line.split()
            if len(parts) >= 6:  # Valid atom line should have at least 6 columns
                if atom_count < len(predicted_coords):
                    # Replace X, Y, Z coordinates (columns 2, 3, 4)
                    parts[2] = f"{predicted_coords[atom_count][0]:.4f}"
                    parts[3] = f"{predicted_coords[atom_count][1]:.4f}"
                    parts[4] = f"{predicted_coords[atom_count][2]:.4f}"
                    
                    # Reconstruct line maintaining MOL2 format
                    # MOL2 format: atom_id atom_name x y z atom_type subst_id subst_name charge
                    new_line = f"{parts[0]:>7} {parts[1]:<8} {parts[2]:>11} {parts[3]:>11} {parts[4]:>11} {parts[5]:<8}"
                    if len(parts) > 6:
                        new_line += f" {parts[6]:>4}"
                    if len(parts) > 7:
                        new_line += f" {parts[7]:<8}"
                    if len(parts) > 8:
                        new_line += f" {parts[8]:>8}"
                    new_line += "\n"
                    
                    modified_lines.append(new_line)
                    atom_count += 1
                else:
                    # More atoms in file than predicted coordinates
                    print(f"Warning: More atoms in MOL2 file than predicted coordinates for {name}")
                    modified_lines.append(line)
            else:
                # Invalid atom line format
                modified_lines.append(line)
        else:
            # Not an atom line - keep original
            modified_lines.append(line)
    
    # Check if we used all predicted coordinates
    if atom_count < len(predicted_coords):
        print(f"Warning: More predicted coordinates than atoms in MOL2 file for {name}")
    
    # Write modified MOL2 file
    with open(output_path, 'w') as f:
        f.writelines(modified_lines)
    
    print(f"Saved predicted MOL2: {output_path}")
    return True


class CoordinateSaverCallback(pl.Callback):
    def __init__(self, 
                 save_every_n_epochs=20, 
                 original_mol2_dir="/home/j2ho/DB/biolip/BioLiP_updated_set/ligand_mol2",
                 output_mol2_dir="./predicted_mol2",
                 save_coords_pt=True,
                 separate_epoch_dirs=True):  # NEW: Control directory structure
        self.save_every_n_epochs = save_every_n_epochs
        self.original_mol2_dir = original_mol2_dir
        self.output_mol2_dir = output_mol2_dir
        self.save_coords_pt = save_coords_pt
        self.separate_epoch_dirs = separate_epoch_dirs
        
        # Create output directories
        os.makedirs(output_mol2_dir, exist_ok=True)
        if save_coords_pt:
            os.makedirs("./predicted_coords", exist_ok=True)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.save_every_n_epochs == 0:
            if pl_module.predict_str:  # Only if structure prediction is enabled
                self.save_coordinates(trainer, pl_module)
    
    def save_coordinates(self, trainer, pl_module):
        """Save coordinates both as MOL2 files and as PyTorch tensors"""
        # Get validation dataloader
        val_dataloader = trainer.val_dataloaders
        
        pl_module.eval()
        all_predictions = []
        all_targets = []
        all_names = []
        saved_mol2_count = 0
        
        # Determine output directory for this epoch
        if self.separate_epoch_dirs:
            # Option 1: Separate directories per epoch
            epoch_output_dir = os.path.join(self.output_mol2_dir, f"epoch_{trainer.current_epoch+1}")
        else:
            # Option 2: Single directory with epoch suffix in filename
            epoch_output_dir = self.output_mol2_dir
        
        with torch.no_grad():
            for batch in val_dataloader:
                results = pl_module(batch)
                
                if 'predicted_ligand_coords' in results:
                    pred_coords = results['predicted_ligand_coords']  # [B, max_atoms, 3]
                    target_coords = results['target_ligand_coords']   # [B, max_atoms, 3]
                    coord_mask = results['coord_target_mask']         # [B, max_atoms]
                    
                    # Extract per-molecule coordinates
                    batch_size = pred_coords.size(0)
                    for b in range(batch_size):
                        mask = coord_mask[b]
                        num_atoms = mask.sum().item()
                        
                        if num_atoms > 0:
                            pred_mol = pred_coords[b][:num_atoms]  # [num_atoms, 3]
                            target_mol = target_coords[b][:num_atoms]  # [num_atoms, 3]
                            
                            # Get molecule name
                            if 'name' in batch:
                                mol_name = batch['name'][b]
                            else:
                                mol_name = f"mol_{len(all_predictions)}"
                            
                            # Save as MOL2 file (with epoch-specific naming)
                            success = save_predicted_mol2_with_epoch(
                                name=mol_name,
                                predicted_coords=pred_mol,
                                original_mol2_dir=self.original_mol2_dir,
                                output_mol2_dir=epoch_output_dir,
                                epoch=trainer.current_epoch + 1,
                                separate_dirs=self.separate_epoch_dirs
                            )
                            
                            if success:
                                saved_mol2_count += 1
                            
                            # Store for tensor saving
                            all_predictions.append(pred_mol.cpu())
                            all_targets.append(target_mol.cpu())
                            all_names.append(mol_name)
        
        # Save as PyTorch tensors (optional)
        if self.save_coords_pt and all_predictions:
            save_path = f"./predicted_coords/epoch_{trainer.current_epoch+1}_coords.pt"
            torch.save({
                'epoch': trainer.current_epoch + 1,
                'predicted_coords': all_predictions,
                'target_coords': all_targets,
                'molecule_names': all_names
            }, save_path)
            
            print(f"Saved {len(all_predictions)} coordinate tensors to {save_path}")
        
        print(f"Saved {saved_mol2_count} MOL2 files for epoch {trainer.current_epoch+1}")


def save_predicted_mol2_with_epoch(name, predicted_coords, original_mol2_dir, output_mol2_dir, epoch, separate_dirs=True):
    """
    Replace coordinates in original MOL2 file with predicted coordinates
    
    Args:
        name: Molecule name (e.g., "5e4t_MBT_A_1")
        predicted_coords: [num_atoms, 3] tensor of predicted coordinates
        original_mol2_dir: Directory containing original MOL2 files
        output_mol2_dir: Directory to save predicted MOL2 files
        epoch: Current epoch number
        separate_dirs: If True, use separate directories; if False, use filename suffix
    """
    # Construct file paths
    original_path = os.path.join(original_mol2_dir, f"{name}.qh.mol2")
    
    if separate_dirs:
        # Option 1: Separate directories per epoch
        # predicted_mol2/epoch_20/5e4t_MBT_A_1.mol2
        # predicted_mol2/epoch_40/5e4t_MBT_A_1.mol2
        output_path = os.path.join(output_mol2_dir, f"{name}.mol2")
    else:
        # Option 2: Single directory with epoch in filename
        # predicted_mol2/5e4t_MBT_A_1_epoch20.mol2
        # predicted_mol2/5e4t_MBT_A_1_epoch40.mol2
        output_path = os.path.join(output_mol2_dir, f"{name}_e{epoch}.mol2")
    
    # Ensure output directory exists
    os.makedirs(output_mol2_dir, exist_ok=True)
    
    # Check if original file exists
    if not os.path.exists(original_path):
        print(f"Warning: Original MOL2 file not found: {original_path}")
        return False
    
    # Convert predicted coordinates to CPU numpy if needed
    if isinstance(predicted_coords, torch.Tensor):
        predicted_coords = predicted_coords.cpu().numpy()
    
    # Read original MOL2 file
    with open(original_path, 'r') as f:
        lines = f.readlines()
    
    # Find @<TRIPOS>ATOM section
    atom_section_start = None
    atom_section_end = None
    
    for i, line in enumerate(lines):
        if line.strip() == "@<TRIPOS>ATOM":
            atom_section_start = i + 1
        elif line.strip().startswith("@<TRIPOS>") and atom_section_start is not None:
            atom_section_end = i
            break
    
    if atom_section_start is None:
        print(f"Warning: Could not find @<TRIPOS>ATOM section in {original_path}")
        return False
    
    # If no end found, assume atoms go until end of file
    if atom_section_end is None:
        atom_section_end = len(lines)
    
    # Replace coordinates in atom lines
    atom_count = 0
    modified_lines = []
    
    for i, line in enumerate(lines):
        if atom_section_start <= i < atom_section_end:
            # This is an atom line - replace coordinates
            parts = line.split()
            if len(parts) >= 6:  # Valid atom line should have at least 6 columns
                if atom_count < len(predicted_coords):
                    # Replace X, Y, Z coordinates (columns 2, 3, 4)
                    parts[2] = f"{predicted_coords[atom_count][0]:.4f}"
                    parts[3] = f"{predicted_coords[atom_count][1]:.4f}"
                    parts[4] = f"{predicted_coords[atom_count][2]:.4f}"
                    
                    # Reconstruct line maintaining MOL2 format
                    # MOL2 format: atom_id atom_name x y z atom_type subst_id subst_name charge
                    new_line = f"{parts[0]:>7} {parts[1]:<8} {parts[2]:>11} {parts[3]:>11} {parts[4]:>11} {parts[5]:<8}"
                    if len(parts) > 6:
                        new_line += f" {parts[6]:>4}"
                    if len(parts) > 7:
                        new_line += f" {parts[7]:<8}"
                    if len(parts) > 8:
                        new_line += f" {parts[8]:>8}"
                    new_line += "\n"
                    
                    modified_lines.append(new_line)
                    atom_count += 1
                else:
                    # More atoms in file than predicted coordinates
                    print(f"Warning: More atoms in MOL2 file than predicted coordinates for {name}")
                    modified_lines.append(line)
            else:
                # Invalid atom line format
                modified_lines.append(line)
        else:
            # Not an atom line - keep original
            modified_lines.append(line)
    
    # Check if we used all predicted coordinates
    if atom_count < len(predicted_coords):
        print(f"Warning: More predicted coordinates than atoms in MOL2 file for {name}")
    
    # Write modified MOL2 file
    with open(output_path, 'w') as f:
        f.writelines(modified_lines)
    
    print(f"Saved predicted MOL2: {output_path}")
    return True