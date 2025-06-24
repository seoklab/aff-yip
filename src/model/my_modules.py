# src/model/my_modules.py
# Adapted for grid-based virtual nodes (equally spaced in pocket)
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def masked_softmax(logits, mask, dim=-1):
    """Softmax with masking"""
    if mask is not None:
        # Set masked positions to large negative value
        logits = logits.masked_fill(~mask, -1e9)
    return F.softmax(logits, dim=dim)

class PairwiseAttentionStructModule(nn.Module):
    def __init__(self, virtual_embed_dim, ligand_embed_dim, hidden_dim=128):
        super().__init__()
        
        # Project embeddings to common dimension for interaction
        self.virtual_projection = nn.Linear(virtual_embed_dim, hidden_dim)
        self.ligand_projection = nn.Linear(ligand_embed_dim, hidden_dim)
        
        # Pairwise interaction layers
        self.pairwise_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Output single score per pair
        )
        
        self.scale = 1.0
        
    def forward(self, virtual_embeddings, virtual_coords, ligand_embeddings, 
                virtual_batch_idx, ligand_batch_idx, target_mask):
        """
        Create pairwise attention between virtual nodes and ligand atoms
        
        Args:
            virtual_embeddings: [N_virtual_total, virtual_embed_dim]
            virtual_coords: [N_virtual_total, 3] - grid coordinates
            ligand_embeddings: [M_ligand_total, ligand_embed_dim] 
            virtual_batch_idx: [N_virtual_total] - batch assignment for virtual nodes
            ligand_batch_idx: [M_ligand_total] - batch assignment for ligand atoms
            target_mask: [B, max_ligand_atoms] - which ligand atoms exist
            
        Returns:
            predicted_coords: [B, max_ligand_atoms, 3]
            attention_weights: [B, max_ligand_atoms, max_virtual_nodes]
        """
        batch_size = target_mask.size(0)
        max_ligand_atoms = target_mask.size(1)
        
        if len(virtual_batch_idx) == 0 or len(ligand_batch_idx) == 0:
            return torch.zeros(batch_size, max_ligand_atoms, 3, device=virtual_embeddings.device), None
        
        # Project to common dimension
        virtual_proj = self.virtual_projection(virtual_embeddings)  # [N, hidden_dim]
        ligand_proj = self.ligand_projection(ligand_embeddings)      # [M, hidden_dim]
        
        predicted_coords = []
        attention_weights_all = []
        
        for b in range(batch_size):
            # Get virtual nodes and ligand atoms for this batch
            v_mask = virtual_batch_idx == b
            l_mask = ligand_batch_idx == b
            
            v_embed_b = virtual_proj[v_mask]     # [N_b, hidden_dim]
            v_coords_b = virtual_coords[v_mask]  # [N_b, 3]
            l_embed_b = ligand_proj[l_mask]      # [M_b, hidden_dim]
            
            N_b = v_embed_b.size(0)  # Number of virtual nodes in batch b
            M_b = l_embed_b.size(0)  # Number of ligand atoms in batch b
            
            if N_b == 0 or M_b == 0:
                predicted_coords.append(torch.zeros(max_ligand_atoms, 3, device=virtual_embeddings.device))
                attention_weights_all.append(torch.zeros(max_ligand_atoms, N_b if N_b > 0 else 1, device=virtual_embeddings.device))
                continue
            
            # Create pairwise features: [M_b, N_b, hidden_dim*2]
            # Each ligand atom interacts with each virtual node
            v_expand = v_embed_b.unsqueeze(0).expand(M_b, -1, -1)  # [M_b, N_b, hidden_dim]
            l_expand = l_embed_b.unsqueeze(1).expand(-1, N_b, -1)  # [M_b, N_b, hidden_dim]
            
            pairwise_features = torch.cat([l_expand, v_expand], dim=-1)  # [M_b, N_b, hidden_dim*2]
            
            # Compute pairwise attention scores: [M_b, N_b]
            z = self.pairwise_mlp(pairwise_features).squeeze(-1)  # [M_b, N_b]
            
            # Create mask for valid virtual nodes (all are valid in this case)
            z_mask = torch.ones(M_b, N_b, dtype=torch.bool, device=virtual_embeddings.device)
            
            # Apply masked softmax to get attention weights
            attention_weights_b = masked_softmax(self.scale * z, mask=z_mask, dim=1)  # [M_b, N_b]
            
            # Compute predicted coordinates using weighted sum
            # attention_weights_b: [M_b, N_b], v_coords_b: [N_b, 3]
            predicted_coords_b = torch.matmul(attention_weights_b, v_coords_b)  # [M_b, 3]
            
            # Pad to max_ligand_atoms
            padded_coords = torch.zeros(max_ligand_atoms, 3, device=virtual_embeddings.device)
            padded_coords[:M_b] = predicted_coords_b
            predicted_coords.append(padded_coords)
            
            # Pad attention weights to max virtual nodes for this batch
            max_virtual_global = max(att.size(-1) if len(attention_weights_all) > 0 else N_b for att in attention_weights_all) if attention_weights_all else N_b
            max_virtual_global = max(max_virtual_global, N_b)
            
            padded_attention = torch.zeros(max_ligand_atoms, max_virtual_global, device=virtual_embeddings.device)
            padded_attention[:M_b, :N_b] = attention_weights_b
            attention_weights_all.append(padded_attention)
        
        predicted_coords = torch.stack(predicted_coords)  # [B, max_ligand_atoms, 3]
        
        # Ensure all attention weights have the same shape
        if attention_weights_all:
            max_virtual_final = max(att.size(-1) for att in attention_weights_all)
            final_attention = []
            for att in attention_weights_all:
                if att.size(-1) < max_virtual_final:
                    padded = torch.zeros(att.size(0), max_virtual_final, device=virtual_embeddings.device)
                    padded[:, :att.size(-1)] = att
                    final_attention.append(padded)
                else:
                    final_attention.append(att)
            attention_weights_all = torch.stack(final_attention)
        else:
            attention_weights_all = torch.zeros(batch_size, max_ligand_atoms, 1, device=virtual_embeddings.device)
        
        return predicted_coords, attention_weights_all

def masked_softmax_2(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes a numerically stable masked softmax.

    Args:
        x: The input tensor (logits).
        mask: A boolean tensor where `False` indicates positions to mask.
        dim: The dimension along which to apply softmax.

    Returns:
        The tensor with softmax applied, where masked elements are zero.
    """
    x_masked = x.masked_fill(~mask, -float('inf'))
    return torch.softmax(x_masked, dim=dim)


class VectorizedPairwiseAttentionStructModule(nn.Module):
    def __init__(self, virtual_embed_dim, ligand_embed_dim, hidden_dim=128):
        super().__init__()
        
        # Project embeddings to a common dimension for interaction
        self.virtual_projection = nn.Linear(virtual_embed_dim, hidden_dim)
        self.ligand_projection = nn.Linear(ligand_embed_dim, hidden_dim)
        
        # Pairwise interaction layers
        self.pairwise_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Output a single score per pair
        )
        
        self.scale = 1.0
        
    def forward(self, virtual_embeddings, virtual_coords, ligand_embeddings,
                virtual_batch_idx, ligand_batch_idx, target_mask):
        """
        Creates pairwise attention between virtual nodes and ligand atoms in a fully vectorized manner.
        
        Args:
            virtual_embeddings: [N_virtual_total, virtual_embed_dim]
            virtual_coords: [N_virtual_total, 3] - grid coordinates
            ligand_embeddings: [M_ligand_total, ligand_embed_dim] 
            virtual_batch_idx: [N_virtual_total] - batch assignment for virtual nodes
            ligand_batch_idx: [M_ligand_total] - batch assignment for ligand atoms
            target_mask: [B, max_ligand_atoms] - which ligand atoms exist
            
        Returns:
            predicted_coords: [B, max_ligand_atoms, 3]
            attention_weights: [B, max_ligand_atoms, max_virtual_nodes_in_batch]
        """
        batch_size = target_mask.size(0)
        max_ligand_atoms = target_mask.size(1)
        device = virtual_embeddings.device

        # Handle empty batches gracefully
        if len(virtual_batch_idx) == 0 or len(ligand_batch_idx) == 0:
            v_counts = torch.bincount(virtual_batch_idx, minlength=batch_size)
            max_virtual_nodes = v_counts.max().item() if v_counts.numel() > 0 else 1
            return torch.zeros(batch_size, max_ligand_atoms, 3, device=device), \
                   torch.zeros(batch_size, max_ligand_atoms, max_virtual_nodes, device=device)
        
        # Project to common dimension (already vectorized)
        virtual_proj = self.virtual_projection(virtual_embeddings)  # [N_total, D]
        ligand_proj = self.ligand_projection(ligand_embeddings)    # [M_total, D]
        
        M_total = ligand_proj.shape[0]
        N_total = virtual_proj.shape[0]

        # 1. Create a mask to identify all valid pairs (ligand atom and virtual node in the same batch item).
        # This creates a boolean tensor of shape [M_total, N_total].
        valid_pair_mask = ligand_batch_idx.unsqueeze(1) == virtual_batch_idx.unsqueeze(0)

        # 2. Get the flat indices of the valid pairs.
        # l_pair_idx and v_pair_idx will have shape [P], where P is the total number of valid pairs in the batch.
        l_pair_idx, v_pair_idx = valid_pair_mask.nonzero(as_tuple=True)
        
        # 3. Gather the embeddings for each valid pair. This is an advanced indexing operation.
        l_embed_pairs = ligand_proj[l_pair_idx]  # [P, D]
        v_embed_pairs = virtual_proj[v_pair_idx]  # [P, D]

        # 4. Create pairwise features and compute attention scores for all pairs in one go.
        pairwise_features = torch.cat([l_embed_pairs, v_embed_pairs], dim=-1)  # [P, D*2]
        pairwise_scores = self.pairwise_mlp(pairwise_features).squeeze(-1)      # [P]

        # 5. Compute attention weights using the dense `valid_pair_mask`.
        # We fill a dense score matrix with our scores, using -inf for invalid pairs.
        # This allows a standard softmax to correctly compute weights for each ligand atom over its corresponding virtual nodes.
        z = torch.full((M_total, N_total), -float('inf'), device=device, dtype=torch.float32)
        z[l_pair_idx, v_pair_idx] = self.scale * pairwise_scores  # [M_total, N_total]
        attention_weights_flat = torch.softmax(z, dim=1)         # [M_total, N_total]
        
        # 6. Compute predicted coordinates as a single weighted average (matrix multiplication).
        predicted_coords_flat = torch.matmul(attention_weights_flat, virtual_coords) # [M_total, 3]
        
        # 7. "Un-batch" the flat results into a padded tensor for coordinates.
        predicted_coords = torch.zeros(batch_size, max_ligand_atoms, 3, device=device)
        l_counts = torch.bincount(ligand_batch_idx, minlength=batch_size)
        l_offsets = torch.zeros_like(l_counts)
        l_offsets[1:] = torch.cumsum(l_counts[:-1], dim=0)
        l_idx_in_batch = torch.arange(M_total, device=device) - l_offsets[ligand_batch_idx]
        
        predicted_coords[ligand_batch_idx, l_idx_in_batch] = predicted_coords_flat

        # 8. Un-batch the flat attention weights into a padded tensor to match the original signature.
        v_counts = torch.bincount(virtual_batch_idx, minlength=batch_size)
        max_virtual_nodes = v_counts.max().item() if v_counts.numel() > 0 else 0
        attention_weights_padded = torch.zeros(batch_size, max_ligand_atoms, max_virtual_nodes, device=device)
        
        if max_virtual_nodes > 0:
            v_offsets = torch.zeros_like(v_counts)
            v_offsets[1:] = torch.cumsum(v_counts[:-1], dim=0)
            v_idx_in_batch = torch.arange(N_total, device=device) - v_offsets[virtual_batch_idx]
            
            # Use advanced indexing to scatter the attention weights from the flat [P] representation
            # into the final padded tensor [B, max_L, max_V].
            batch_indices_for_pairs = ligand_batch_idx[l_pair_idx]
            ligand_indices_for_pairs = l_idx_in_batch[l_pair_idx]
            virtual_indices_for_pairs = v_idx_in_batch[v_pair_idx]
            
            attention_weights_padded[batch_indices_for_pairs, ligand_indices_for_pairs, virtual_indices_for_pairs] = attention_weights_flat[l_pair_idx, v_pair_idx]
        
        return predicted_coords, attention_weights_padded
