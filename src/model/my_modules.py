# Adapted for grid-based virtual nodes (equally spaced in pocket)
import torch
import torch.nn as nn

class GridBasedStructModule(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        # Project virtual node embeddings to attention weights
        self.virtual_to_weights = nn.Linear(embed_dim, num_heads)
        self.num_heads = num_heads
        self.scale = 1.0
        
    def forward(self, virtual_embeddings, virtual_coords, target_mask, batch_idx):
        """
        Predict coordinates using grid-based virtual nodes
        
        Args:
            virtual_embeddings: [N_virtual_total, embed_dim] - learned features for grid points
            virtual_coords: [N_virtual_total, 3] - fixed grid coordinates in pocket
            target_mask: [B, max_ligand_atoms] - which ligand atoms exist per batch
            batch_idx: [N_virtual_total] - batch assignment for virtual nodes
            
        Returns:
            predicted_coords: [B, max_ligand_atoms, 3] - predicted ligand coordinates
            attention_weights: [B, max_ligand_atoms, max_virtual_nodes] - attention maps
        """
        
        batch_size = target_mask.size(0)
        max_ligand_atoms = target_mask.size(1)
        max_virtual_nodes = virtual_embeddings.size(0) // batch_size  # Assuming equal grid per sample
        
        predicted_coords = []
        attention_weights_all = []
        
        for b in range(batch_size):
            # Get virtual nodes for this batch
            v_mask = batch_idx == b
            v_embed = virtual_embeddings[v_mask]  # [N_grid, embed_dim]
            v_coords = virtual_coords[v_mask]     # [N_grid, 3] - grid positions
            
            # Number of ligand atoms in this sample
            num_ligand_atoms = target_mask[b].sum().item()
            
            if num_ligand_atoms == 0:
                # No ligand atoms in this sample
                predicted_coords.append(torch.zeros(max_ligand_atoms, 3, device=virtual_embeddings.device))
                attention_weights_all.append(torch.zeros(max_ligand_atoms, v_embed.size(0), device=virtual_embeddings.device))
                continue
            
            # For each ligand atom position we need to predict
            sample_coords = []
            sample_weights = []
            
            for atom_idx in range(max_ligand_atoms):
                if target_mask[b, atom_idx]:
                    # This ligand atom exists - predict its coordinate
                    
                    # Method 1: Global attention (your original approach)
                    # All virtual nodes contribute equally to this ligand atom
                    weights = self.virtual_to_weights(v_embed)[:, 0]  # [N_grid]
                    weights = torch.softmax(self.scale * weights, dim=0)  # [N_grid]
                    
                    # Method 2: Position-aware attention (alternative)
                    # MSK 비슷하게 추가 가능... 근데 무거워지니까 굳이? at least not for aff pred purposes
                    
                    # Weighted sum of grid coordinates
                    pred_coord = torch.sum(weights.unsqueeze(-1) * v_coords, dim=0)  # [3]
                    sample_coords.append(pred_coord)
                    sample_weights.append(weights)
                    
                else:
                    # This ligand atom doesn't exist - pad with zeros
                    sample_coords.append(torch.zeros(3, device=virtual_embeddings.device))
                    sample_weights.append(torch.zeros(v_embed.size(0), device=virtual_embeddings.device))
            
            predicted_coords.append(torch.stack(sample_coords))  # [max_ligand_atoms, 3]
            attention_weights_all.append(torch.stack(sample_weights))  # [max_ligand_atoms, N_grid]
        
        predicted_coords = torch.stack(predicted_coords)  # [B, max_ligand_atoms, 3]
        attention_weights_all = torch.stack(attention_weights_all)  # [B, max_ligand_atoms, N_grid]
        
        return predicted_coords, attention_weights_all

# Alternative version that pads attention weights to same size
class GridBasedStructModulePadded(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.virtual_to_weights = nn.Linear(embed_dim, num_heads)
        self.num_heads = num_heads
        self.scale = 1.0
        
    def forward(self, virtual_embeddings, virtual_coords, target_mask, batch_idx):
        """
        Version that pads attention weights to enable stacking
        """
        batch_size = target_mask.size(0)
        max_ligand_atoms = target_mask.size(1)

        if len(batch_idx) == 0:
            # No virtual nodes, return empty tensors
            predicted_coords = torch.zeros(batch_size, max_ligand_atoms, 3, device=virtual_embeddings.device)
            attention_weights_all = torch.zeros(batch_size, max_ligand_atoms, 1, device=virtual_embeddings.device)
            return None, None 

        # Find max number of virtual nodes across all samples
        virtual_counts = torch.bincount(batch_idx)
        max_virtual_nodes = virtual_counts.max().item()
        
        predicted_coords = []
        attention_weights_all = []
        
        for b in range(batch_size):
            # Get virtual nodes for this batch
            v_mask = batch_idx == b
            v_embed = virtual_embeddings[v_mask]  # [N_grid_b, embed_dim]
            v_coords = virtual_coords[v_mask]     # [N_grid_b, 3]
            
            num_virtual_this_batch = v_embed.size(0)
            
            if target_mask[b].sum().item() == 0 or num_virtual_this_batch == 0:
                # No ligand atoms or virtual nodes
                predicted_coords.append(torch.zeros(max_ligand_atoms, 3, device=virtual_embeddings.device))
                attention_weights_all.append(torch.zeros(max_ligand_atoms, max_virtual_nodes, device=virtual_embeddings.device))
                continue
            
            sample_coords = []
            sample_weights = []
            
            for atom_idx in range(max_ligand_atoms):
                if target_mask[b, atom_idx]:
                    # Predict coordinate for this atom
                    weights = self.virtual_to_weights(v_embed)[:, 0]  # [N_grid_b]
                    weights = torch.softmax(self.scale * weights, dim=0)  # [N_grid_b]
                    
                    pred_coord = torch.sum(weights.unsqueeze(-1) * v_coords, dim=0)  # [3]
                    sample_coords.append(pred_coord)
                    
                    # Pad weights to max_virtual_nodes
                    padded_weights = torch.zeros(max_virtual_nodes, device=virtual_embeddings.device)
                    padded_weights[:num_virtual_this_batch] = weights
                    sample_weights.append(padded_weights)
                else:
                    # Padding
                    sample_coords.append(torch.zeros(3, device=virtual_embeddings.device))
                    sample_weights.append(torch.zeros(max_virtual_nodes, device=virtual_embeddings.device))
            
            predicted_coords.append(torch.stack(sample_coords))  # [max_ligand_atoms, 3]
            attention_weights_all.append(torch.stack(sample_weights))  # [max_ligand_atoms, max_virtual_nodes]
        
        predicted_coords = torch.stack(predicted_coords)  # [B, max_ligand_atoms, 3]
        attention_weights_all = torch.stack(attention_weights_all)  # [B, max_ligand_atoms, max_virtual_nodes]
        
        return predicted_coords, attention_weights_all
