# src/model/my_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .structure_modules.egnn_layers import EGNNConv, CoordEGNNConv

class DirectCoordinatePredictor(nn.Module):
    """
    MLP based
    most simple coordinate predictor.
    """
    
    def __init__(self, 
                 lig_embed_dim=196,
                 prot_embed_dim=196,
                 hidden_dim=128,
                 max_sidechain_atoms=10,  # Maximum atoms in any sidechain. 10
                 dropout=0.1):
        super().__init__()
        
        self.max_sidechain_atoms = max_sidechain_atoms
        
        # ligand coordinate prediction from ligand embeddings
        self.ligand_predictor = nn.Sequential(
            nn.Linear(lig_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        # Predict all sidechain atom coordinates at once
        self.sidechain_predictor = nn.Sequential(
            nn.Linear(prot_embed_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_sidechain_atoms * 3)  # 3 coords per atom
        )
        
    def forward(self,
                ligand_embeddings,      # [N_ligand, embed_dim]
                ligand_batch_idx,       # [N_ligand] 
                protein_embeddings,     # [N_protein, embed_dim]
                protein_batch_idx,      # [N_protein]
                target_mask,            # [B, max_ligand_atoms]
                X_sidechain_mask,       # [B, N_prot, N_sidechain_max]
                protein_mask,
                **kwargs):              # Ignore other inputs for compatibility
        
        lig_batch_size = target_mask.size(0)
        max_ligand_atoms = target_mask.size(1)
        num_residues = protein_mask.size(1) 
        max_sidechain_atoms = X_sidechain_mask.size(-1)
        
        # ===== LIGAND PREDICTION =====
        pred_ligand_coords = torch.zeros(
            lig_batch_size, max_ligand_atoms, 3,
            device=ligand_embeddings.device
        )
        
        # Predict coordinates directly from ligand embeddings
        ligand_coords_raw = self.ligand_predictor(ligand_embeddings)  # [N_ligand, 3]
        
        # Organize by batch
        coord_idx = 0
        for b in range(lig_batch_size):
            num_atoms = (ligand_batch_idx == b).sum().item()
            if num_atoms > 0:
                pred_ligand_coords[b, :num_atoms] = ligand_coords_raw[coord_idx:coord_idx + num_atoms]
                coord_idx += num_atoms
        
        # ===== SIDECHAIN PREDICTION (AA)=====
        prot_batch_size = lig_batch_size
        pred_sidechain_coords = torch.zeros(
            prot_batch_size, num_residues, max_sidechain_atoms, 3,
            device=protein_embeddings.device
        ) # [B, max_residues, N_sidechain_max, 3] 
        # Predict ALL sidechain atom coordinates at once
        protein_coords_raw = self.sidechain_predictor(protein_embeddings)  # [N_protein, max_sidechain_atoms * 3]
        protein_coords_reshaped = protein_coords_raw.view(-1, max_sidechain_atoms, 3)  # [N_protein, max_sidechain_atoms, 3]
        # Organize by batch and residue (similar to ligand organization)
        coord_idx = 0
        for b in range(prot_batch_size):
            # Count how many protein residues belong to this batch
            num_residues_in_batch = (protein_batch_idx == b).sum().item()
            if num_residues_in_batch > 0:
                # Assign coordinates for all residues in this batch
                batch_coords = protein_coords_reshaped[coord_idx:coord_idx + num_residues_in_batch]  # [num_residues_in_batch, max_sidechain_atoms, 3]
                pred_sidechain_coords[b, :num_residues_in_batch] = batch_coords
                coord_idx += num_residues_in_batch
        pred_sidechain_coords = pred_sidechain_coords * protein_mask.unsqueeze(-1).unsqueeze(-1)  # [B, max_residues, 1, 1]
        pred_sidechain_coords = pred_sidechain_coords * X_sidechain_mask.unsqueeze(-1)  # [B, N_prot, N_sidechain_max, 1]
        return pred_ligand_coords, pred_sidechain_coords

