import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean, scatter_add
from .egnn_layers import EGNNConv, CoordEGNNConv, CoordEGNNConv_v2, CoordEGNNConv_v3
import numpy as np

class EGNNCoordinatePredictor(nn.Module):
    """
    EGNN-based coordinate predictor that treats sidechain atoms as graph nodes
    and uses existing ligand graph structure.
    """
    
    def __init__(self,
                 lig_embed_dim=196,
                 prot_embed_dim=196,
                 hidden_dim=128,
                 num_layers=3,
                 max_sidechain_atoms=10,
                 dropout=0.1,
                 coord_update_type="v2",
                 pocket_radius=8.0):
        super().__init__()
        
        self.max_sidechain_atoms = max_sidechain_atoms
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.coord_update_type = coord_update_type
        self.pocket_radius = pocket_radius
        
        # Feature projection layers
        self.ligand_proj = nn.Linear(lig_embed_dim, hidden_dim)
        self.protein_proj = nn.Linear(prot_embed_dim, hidden_dim)
        self.sidechain_proj = nn.Linear(64, hidden_dim)  # For sidechain atom features
        
        # EGNN layers for joint ligand-protein-sidechain processing
        self.egnn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EGNNConv(
                in_size=hidden_dim,
                hidden_size=hidden_dim,
                out_size=hidden_dim,
                dropout=dropout,
                edge_feat_size=7  # 1 distance + 6 edge type features
            )
            self.egnn_layers.append(layer)
        
        # Coordinate update layers (using the variants we ported)
        self.coord_update_layers = nn.ModuleList()
        for _ in range(num_layers):
            if coord_update_type == "v1":
                coord_layer = CoordEGNNConv(
                    in_size=hidden_dim,
                    hidden_size=hidden_dim,
                    dropout=dropout,
                    edge_feat_size=7  # 1 distance + 6 edge type features
                )
            elif coord_update_type == "v2":
                from .egnn_layers import CoordEGNNConv_v2
                coord_layer = CoordEGNNConv_v2(
                    in_size=hidden_dim,
                    hidden_size=hidden_dim,
                    dropout=dropout,
                    edge_feat_size=7  # 1 distance + 6 edge type features
                )
            elif coord_update_type == "v3":
                from .egnn_layers import CoordEGNNConv_v3
                coord_layer = CoordEGNNConv_v3(
                    in_size=hidden_dim,
                    hidden_size=hidden_dim,
                    dropout=dropout,
                    edge_feat_size=7  # 1 distance + 6 edge type features
                )
            else:
                raise ValueError(f"Unknown coord_update_type: {coord_update_type}")
            
            self.coord_update_layers.append(coord_layer)
        
        # Node type embeddings
        self.node_type_embedding = nn.Embedding(4, hidden_dim)  # 0:ligand, 1:backbone, 2:sidechain, 3:virtual
        
        # Sidechain atom type embeddings (simplified)
        self.sidechain_atom_types = ['C', 'N', 'O', 'S', 'P', 'UNK']
        self.sidechain_atom_embedding = nn.Embedding(len(self.sidechain_atom_types), 32)
        
    def _create_sidechain_nodes(self, protein_embeddings, protein_batch_idx, backbone_coords, 
                               X_sidechain_mask, sidechain_map):
        """
        Create sidechain atom nodes using actual sidechain info from protein structure
        """
        device = protein_embeddings.device
        batch_size = X_sidechain_mask.size(0)
        max_residues = X_sidechain_mask.size(1)
        max_atoms = X_sidechain_mask.size(2)
        
        # Handle sidechain_map - it could be a single dict or tuple/list of dicts for batched data
        if isinstance(sidechain_map, (tuple, list)):
            # Batched case: sidechain_map is (dict1, dict2, ...)
            sidechain_maps = sidechain_map
        else:
            # Single case: wrap in list for consistent handling
            sidechain_maps = [sidechain_map]
        
        # Create features and coordinates for all sidechain atoms
        sidechain_features = []
        sidechain_coords = []
        sidechain_batch_idx = []
        sidechain_residue_idx = []
        sidechain_atom_names = []
        
        # Map atom names to simplified types (based on first letter for element)
        def get_atom_type_idx(atom_name):
            if atom_name.startswith('C'):
                return 0  # Carbon
            elif atom_name.startswith('N'):
                return 1  # Nitrogen
            elif atom_name.startswith('O'):
                return 2  # Oxygen
            elif atom_name.startswith('S'):
                return 3  # Sulfur
            elif atom_name.startswith('P'):
                return 4  # Phosphorus
            else:
                return 5  # Unknown
        
        def parse_residue_key(key):
            # Remove parentheses and split by dots
            # '(GLN.A.68)' -> ['GLN', 'A', '68']
            clean_key = key.strip('()')
            parts = clean_key.split('.')
            if len(parts) >= 3:
                res_name = parts[0]
                chain_id = parts[1]
                res_num = parts[2]
                return res_name, chain_id, res_num
            return None, None, None
        
        protein_res_idx = 0
        for batch_id in range(batch_size):
            # Find protein residues in this batch
            batch_protein_mask = protein_batch_idx == batch_id
            num_residues_in_batch = batch_protein_mask.sum().item()
            
            if num_residues_in_batch == 0:
                continue
            
            # Get the sidechain map for this batch
            if batch_id < len(sidechain_maps):
                batch_sidechain_map = sidechain_maps[batch_id]
                residue_keys = list(batch_sidechain_map.keys())
            else:
                # No sidechain map for this batch
                protein_res_idx += num_residues_in_batch
                continue
                
            # Get backbone coordinates for this batch
            batch_backbone_coords = backbone_coords[batch_protein_mask]
            
            for res_idx in range(num_residues_in_batch):
                global_res_idx = protein_res_idx + res_idx
                
                # Find corresponding residue in sidechain_map using res_idx (batch-local index)
                if res_idx < len(residue_keys):
                    residue_key = residue_keys[res_idx]
                    res_name, chain_id, res_num = parse_residue_key(residue_key)
                    
                    # Get sidechain info for this residue from the sidechain_map
                    if residue_key in batch_sidechain_map:
                        sidechain_atom_coords = batch_sidechain_map[residue_key]  # Dict of {atom_name: coordinates}
                        
                        # Create nodes for each sidechain atom that actually exists
                        for atom_idx, (atom_name, atom_coords) in enumerate(sidechain_atom_coords.items()):
                            if atom_idx >= max_atoms:
                                break
                                
                            # Check if this atom position is valid in the mask
                            if res_idx < X_sidechain_mask.size(1) and atom_idx < X_sidechain_mask.size(2):
                                if not X_sidechain_mask[batch_id, res_idx, atom_idx]:
                                    continue
                            
                            # Create atom features
                            atom_type_idx = get_atom_type_idx(atom_name)
                            atom_type_emb = self.sidechain_atom_embedding(torch.tensor(atom_type_idx, device=device))
                            
                            # Use residue embedding as base feature
                            residue_emb = protein_embeddings[global_res_idx]
                            
                            # Combine features
                            atom_features = torch.cat([residue_emb[:32], atom_type_emb], dim=0)  # Take first 32 dims + atom type
                            sidechain_features.append(atom_features)
                            
                            # Use random initial position near CA (don't use ground truth coords)
                            ca_coord = batch_backbone_coords[res_idx]
                            # Random offset in sphere around CA (1-3 Angstroms)
                            random_offset = torch.randn(3, device=device) * 0.5
                            random_offset = random_offset / random_offset.norm() * (1.0 + torch.rand(1, device=device) * 2.0)
                            initial_coord = ca_coord + random_offset
                            sidechain_coords.append(initial_coord)
                            
                            sidechain_batch_idx.append(batch_id)
                            sidechain_residue_idx.append(global_res_idx)
                            sidechain_atom_names.append(atom_name)
            
            protein_res_idx += num_residues_in_batch
        
        if not sidechain_features:
            return (torch.empty(0, 64, device=device), 
                    torch.empty(0, 3, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    [])
        
        sidechain_features = torch.stack(sidechain_features)
        sidechain_coords = torch.stack(sidechain_coords)
        sidechain_batch_idx = torch.tensor(sidechain_batch_idx, device=device)
        sidechain_residue_idx = torch.tensor(sidechain_residue_idx, device=device)
        
        return sidechain_features, sidechain_coords, sidechain_batch_idx, sidechain_residue_idx, sidechain_atom_names
    
    def _build_combined_graph(self, ligand_batch_idx, ligand_coords, ligand_edge_index,
                             protein_batch_idx, backbone_coords, 
                             sidechain_batch_idx, sidechain_coords, sidechain_residue_idx, sidechain_atom_names):
        """
        Build combined graph with specific edge formation rules:
        - Protein-ligand edges within 10Å from CB atoms (max 20 per ligand)
        - Intra-ligand edges within 5Å 
        - Mandatory intra-residue connections
        - Edge types: lig-lig, protein_N-lig, protein_C-lig, protein_O-lig, others
        """
        device = ligand_coords.device
        all_coords = []
        all_batch_idx = []
        node_types = []
        node_elements = []  # Track element types for edge descriptors
        node_offsets = []
        
        # Track node offsets for edge mapping
        current_offset = 0
        
        # Add ligand nodes (type 0)
        if ligand_coords.size(0) > 0:
            all_coords.append(ligand_coords)
            all_batch_idx.append(ligand_batch_idx)
            node_types.append(torch.zeros(ligand_coords.size(0), device=device, dtype=torch.long))
            node_elements.append(torch.zeros(ligand_coords.size(0), device=device, dtype=torch.long))  # Assume C for now
            node_offsets.append(('ligand', current_offset, ligand_coords.size(0)))
            current_offset += ligand_coords.size(0)
        
        # Add backbone nodes (type 1)
        if backbone_coords.size(0) > 0:
            all_coords.append(backbone_coords)
            all_batch_idx.append(protein_batch_idx)
            node_types.append(torch.ones(backbone_coords.size(0), device=device, dtype=torch.long))
            node_elements.append(torch.zeros(backbone_coords.size(0), device=device, dtype=torch.long))  # CA = C
            node_offsets.append(('backbone', current_offset, backbone_coords.size(0)))
            current_offset += backbone_coords.size(0)
        
        # Add sidechain nodes (type 2)
        if sidechain_coords.size(0) > 0:
            all_coords.append(sidechain_coords)
            all_batch_idx.append(sidechain_batch_idx)
            node_types.append(2 * torch.ones(sidechain_coords.size(0), device=device, dtype=torch.long))
            
            # Determine element types for sidechain atoms
            sidechain_elements = []
            for atom_name in sidechain_atom_names:
                if atom_name.startswith('C'):
                    sidechain_elements.append(0)  # Carbon
                elif atom_name.startswith('N'):
                    sidechain_elements.append(1)  # Nitrogen
                elif atom_name.startswith('O'):
                    sidechain_elements.append(2)  # Oxygen
                elif atom_name.startswith('S'):
                    sidechain_elements.append(3)  # Sulfur
                else:
                    sidechain_elements.append(4)  # Others
            node_elements.append(torch.tensor(sidechain_elements, device=device, dtype=torch.long))
            node_offsets.append(('sidechain', current_offset, sidechain_coords.size(0)))
        
        if not all_coords:
            return (torch.empty(0, 3, device=device), 
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(2, 0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device))
        
        # Concatenate all coordinates and batch indices
        combined_coords = torch.cat(all_coords, dim=0)
        combined_batch_idx = torch.cat(all_batch_idx, dim=0)
        combined_node_types = torch.cat(node_types, dim=0)
        combined_node_elements = torch.cat(node_elements, dim=0)
        
        # Build edges separately for each batch
        edge_sources = []
        edge_targets = []
        edge_distances = []
        edge_type_encoding = []
        
        for batch_id in torch.unique(combined_batch_idx):
            batch_mask = combined_batch_idx == batch_id
            batch_coords = combined_coords[batch_mask]
            batch_node_types = combined_node_types[batch_mask]
            batch_node_elements = combined_node_elements[batch_mask]
            batch_indices = torch.where(batch_mask)[0]
            
            if batch_coords.size(0) < 2:
                continue
            
            # Get node indices for this batch
            ligand_mask = batch_node_types == 0
            backbone_mask = batch_node_types == 1
            sidechain_mask = batch_node_types == 2
            
            ligand_indices = batch_indices[ligand_mask]
            backbone_indices = batch_indices[backbone_mask]
            sidechain_indices = batch_indices[sidechain_mask]
            
            # 1. Use existing ligand bonds from featurizer (covalently bonded)
            if ligand_edge_index.size(1) > 0:
                # Filter edges that belong to this batch
                ligand_batch_mask_src = torch.isin(ligand_edge_index[0], ligand_indices)
                ligand_batch_mask_dst = torch.isin(ligand_edge_index[1], ligand_indices)
                ligand_batch_mask = ligand_batch_mask_src & ligand_batch_mask_dst
                
                if ligand_batch_mask.any():
                    valid_edges = ligand_edge_index[:, ligand_batch_mask]
                    edge_sources.extend(valid_edges[0].tolist())
                    edge_targets.extend(valid_edges[1].tolist())
                    
                    # Compute distances and edge types for covalent bonds
                    for i, j in zip(valid_edges[0], valid_edges[1]):
                        dist = torch.norm(combined_coords[i] - combined_coords[j])
                        edge_distances.append(dist)
                        edge_type_encoding.append([1, 0, 0, 0, 0, 0])  # lig-lig (covalent)
            
            # 2. Non-bonded intra-ligand proximity edges within 5Å
            if ligand_indices.size(0) > 1:
                ligand_coords_batch = batch_coords[ligand_mask]
                ligand_dist_matrix = torch.cdist(ligand_coords_batch, ligand_coords_batch)
                
                # Find pairs within 5Å that are NOT already connected by bonds
                close_pairs = torch.where((ligand_dist_matrix < 5.0) & (ligand_dist_matrix > 0))
                for i, j in zip(close_pairs[0], close_pairs[1]):
                    global_i = ligand_indices[i]
                    global_j = ligand_indices[j]
                    
                    # Check if this pair is already connected by existing bonds
                    existing_bond = False
                    if ligand_edge_index.size(1) > 0:
                        existing_bond = ((ligand_edge_index[0] == global_i) & (ligand_edge_index[1] == global_j)).any()
                    
                    if not existing_bond:
                        edge_sources.extend([global_i, global_j])
                        edge_targets.extend([global_j, global_i])
                        
                        dist = ligand_dist_matrix[i, j]
                        edge_distances.extend([dist, dist])
                        edge_type_encoding.extend([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])  # lig-lig (non-bonded)
            
            # 3. Mandatory intra-residue connections (sidechain to backbone)
            if sidechain_indices.size(0) > 0 and backbone_indices.size(0) > 0:
                # Get sidechain residue indices for this batch
                batch_sidechain_mask = torch.isin(torch.arange(len(sidechain_residue_idx), device=device), 
                                                 torch.where(sidechain_batch_idx == batch_id)[0])
                batch_sidechain_residue_idx = sidechain_residue_idx[batch_sidechain_mask]
                
                # Map global residue indices to batch-local backbone indices
                batch_protein_mask = protein_batch_idx == batch_id
                batch_backbone_residue_indices = torch.where(batch_protein_mask)[0]
                
                for sidechain_idx, residue_idx in enumerate(batch_sidechain_residue_idx):
                    # Find corresponding backbone index
                    backbone_local_idx = (batch_backbone_residue_indices == residue_idx).nonzero()
                    if backbone_local_idx.size(0) > 0:
                        backbone_local_idx = backbone_local_idx[0, 0].item()
                        sidechain_global_idx = sidechain_indices[sidechain_idx]
                        backbone_global_idx = backbone_indices[backbone_local_idx]
                        
                        edge_sources.extend([sidechain_global_idx, backbone_global_idx])
                        edge_targets.extend([backbone_global_idx, sidechain_global_idx])
                        
                        dist = torch.norm(combined_coords[sidechain_global_idx] - combined_coords[backbone_global_idx])
                        edge_distances.extend([dist, dist])
                        
                        # Determine edge type based on sidechain atom element
                        sidechain_element = combined_node_elements[sidechain_global_idx]
                        if sidechain_element == 1:  # Nitrogen
                            edge_type = [0, 0, 1, 0, 0, 0]  # protein_N-lig
                        elif sidechain_element == 0:  # Carbon
                            edge_type = [0, 0, 0, 1, 0, 0]  # protein_C-lig
                        elif sidechain_element == 2:  # Oxygen
                            edge_type = [0, 0, 0, 0, 1, 0]  # protein_O-lig
                        else:
                            edge_type = [0, 0, 0, 0, 0, 1]  # others
                        edge_type_encoding.extend([edge_type, edge_type])
            
            # 4. Protein-ligand edges within 10Å (max 20 per ligand, from CB or backbone)
            if ligand_indices.size(0) > 0 and (backbone_indices.size(0) > 0 or sidechain_indices.size(0) > 0):
                ligand_coords_batch = batch_coords[ligand_mask]
                
                # For each ligand atom, find closest protein atoms
                protein_coords_for_distance = []
                protein_indices_for_distance = []
                protein_elements_for_distance = []
                
                # Use CB atoms if available, otherwise backbone
                if sidechain_indices.size(0) > 0:
                    # Get sidechain atoms for this specific batch
                    batch_sidechain_mask = torch.isin(torch.arange(len(sidechain_atom_names), device=device), 
                                                     torch.where(sidechain_batch_idx == batch_id)[0])
                    batch_sidechain_atom_names = [sidechain_atom_names[i] for i in range(len(sidechain_atom_names)) if batch_sidechain_mask[i]]
                    
                    # Find CB atoms within this batch's sidechain atoms
                    batch_cb_mask = torch.tensor([name == 'CB' for name in batch_sidechain_atom_names], device=device)
                    
                    if batch_cb_mask.any():
                        # Get the actual indices of CB atoms within sidechain_indices for this batch
                        batch_sidechain_indices = sidechain_indices[torch.isin(sidechain_indices, batch_indices)]
                        if batch_sidechain_indices.size(0) > 0:
                            # Map CB mask to actual sidechain indices for this batch
                            batch_sidechain_local_indices = []
                            cb_found = []
                            sidechain_idx = 0
                            for i, atom_name in enumerate(sidechain_atom_names):
                                if sidechain_batch_idx[i] == batch_id:
                                    batch_sidechain_local_indices.append(sidechain_idx)
                                    cb_found.append(atom_name == 'CB')
                                    sidechain_idx += 1
                            
                            if cb_found and any(cb_found):
                                cb_indices_local = [batch_sidechain_indices[i] for i, is_cb in enumerate(cb_found) if is_cb]
                                batch_cb_indices = torch.tensor(cb_indices_local, device=device, dtype=torch.long)
                                
                                if batch_cb_indices.size(0) > 0:
                                    cb_coords = combined_coords[batch_cb_indices]
                                    protein_coords_for_distance.append(cb_coords)
                                    protein_indices_for_distance.extend(batch_cb_indices.tolist())
                                    protein_elements_for_distance.extend([0] * batch_cb_indices.size(0))  # CB = Carbon
                
                # Add backbone atoms for residues without CB or as fallback
                if backbone_indices.size(0) > 0:
                    backbone_coords_batch = combined_coords[backbone_indices]
                    protein_coords_for_distance.append(backbone_coords_batch)
                    protein_indices_for_distance.extend(backbone_indices.tolist())
                    protein_elements_for_distance.extend([0] * backbone_indices.size(0))  # CA = Carbon
                
                if protein_coords_for_distance:
                    protein_coords_combined = torch.cat(protein_coords_for_distance, dim=0)
                    
                    # Compute distances between ligand and protein atoms
                    lig_prot_dist_matrix = torch.cdist(ligand_coords_batch, protein_coords_combined)
                    
                    for lig_idx in range(ligand_coords_batch.size(0)):
                        distances_to_protein = lig_prot_dist_matrix[lig_idx]
                        close_protein = torch.where(distances_to_protein < 10.0)[0]
                        
                        if close_protein.size(0) > 0:
                            # Sort by distance and take top 20
                            sorted_indices = torch.argsort(distances_to_protein[close_protein])
                            top_indices = close_protein[sorted_indices[:20]]
                            
                            for prot_local_idx in top_indices:
                                global_lig_idx = ligand_indices[lig_idx]
                                global_prot_idx = protein_indices_for_distance[prot_local_idx]
                                
                                edge_sources.extend([global_lig_idx, global_prot_idx])
                                edge_targets.extend([global_prot_idx, global_lig_idx])
                                
                                dist = distances_to_protein[prot_local_idx]
                                edge_distances.extend([dist, dist])
                                
                                # Determine edge type based on protein atom element
                                prot_element = protein_elements_for_distance[prot_local_idx]
                                if prot_element == 1:  # Nitrogen
                                    edge_type = [0, 0, 1, 0, 0, 0]  # protein_N-lig
                                elif prot_element == 0:  # Carbon
                                    edge_type = [0, 0, 0, 1, 0, 0]  # protein_C-lig
                                elif prot_element == 2:  # Oxygen
                                    edge_type = [0, 0, 0, 0, 1, 0]  # protein_O-lig
                                else:
                                    edge_type = [0, 0, 0, 0, 0, 1]  # others
                                edge_type_encoding.extend([edge_type, edge_type])
        
        if edge_sources:
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long, device=device)
            edge_distances = torch.stack(edge_distances)
            edge_type_encoding = torch.tensor(edge_type_encoding, dtype=torch.float, device=device)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            edge_distances = torch.empty(0, device=device)
            edge_type_encoding = torch.empty(0, 6, device=device)  # 6 edge types now
        
        return combined_coords, combined_batch_idx, edge_index, combined_node_types, edge_distances, edge_type_encoding
    
    def forward(self,
                ligand_embeddings,      # [N_ligand, embed_dim]
                ligand_batch_idx,       # [N_ligand] 
                protein_embeddings,     # [N_protein, embed_dim]
                protein_batch_idx,      # [N_protein]
                target_mask,            # [B, max_ligand_atoms]
                X_sidechain_mask,       # [B, N_prot, N_sidechain_max]
                protein_mask,           # [B, max_residues]
                protein_virtual_batch,  # Original batch object
                **kwargs):              # Additional inputs for compatibility
        
        device = ligand_embeddings.device
        lig_batch_size = target_mask.size(0)
        max_ligand_atoms = target_mask.size(1)
        num_residues = protein_mask.size(1) 
        max_sidechain_atoms = X_sidechain_mask.size(-1)
        
        # Get coordinates from the ligand batch (use existing graph structure)
        ligand_batch = kwargs.get('ligand_batch')
        if ligand_batch is not None:
            ligand_coords = ligand_batch.pos
            ligand_edge_index = ligand_batch.edge_index
        else:
            # Fallback: random initialization
            ligand_coords = torch.randn(ligand_embeddings.size(0), 3, device=device) * 2.0
            ligand_edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        
        # Get backbone coordinates
        protein_mask_1d = protein_virtual_batch.node_type == 0
        backbone_coords = protein_virtual_batch.x[protein_mask_1d]
        
        # Get residue list and sidechain map for sidechain atom creation
        sidechain_map = kwargs.get('sidechain_map', {})
        
        # Create sidechain nodes using actual sidechain info
        sidechain_features, sidechain_coords, sidechain_batch_idx, sidechain_residue_idx, sidechain_atom_names = \
            self._create_sidechain_nodes(protein_embeddings, protein_batch_idx, backbone_coords, 
                                       X_sidechain_mask, sidechain_map)
        
        # Project all features to hidden dimension
        h_ligand = self.ligand_proj(ligand_embeddings) if ligand_embeddings.size(0) > 0 else torch.empty(0, self.hidden_dim, device=device)
        h_protein = self.protein_proj(protein_embeddings) if protein_embeddings.size(0) > 0 else torch.empty(0, self.hidden_dim, device=device)
        h_sidechain = self.sidechain_proj(sidechain_features) if sidechain_features.size(0) > 0 else torch.empty(0, self.hidden_dim, device=device)
        
        # Build combined graph with edge descriptors
        combined_coords, combined_batch_idx, combined_edge_index, combined_node_types, edge_distances, edge_type_encoding = \
            self._build_combined_graph(ligand_batch_idx, ligand_coords, ligand_edge_index,
                                     protein_batch_idx, backbone_coords,
                                     sidechain_batch_idx, sidechain_coords, sidechain_residue_idx, sidechain_atom_names)
        
        # Combine all node features
        all_features = []
        if h_ligand.size(0) > 0:
            all_features.append(h_ligand)
        if h_protein.size(0) > 0:
            all_features.append(h_protein)
        if h_sidechain.size(0) > 0:
            all_features.append(h_sidechain)
        
        if not all_features:
            # Return empty results
            return (torch.zeros(lig_batch_size, max_ligand_atoms, 3, device=device),
                    torch.zeros(lig_batch_size, num_residues, max_sidechain_atoms, 3, device=device))
        
        combined_features = torch.cat(all_features, dim=0)
        
        # Add node type embeddings
        type_embeddings = self.node_type_embedding(combined_node_types)
        combined_features = combined_features + type_embeddings
        
        # Add edge features to node features for EGNN processing
        if combined_edge_index.size(1) > 0:
            # Create edge attributes combining distance and type encoding
            edge_attr = torch.cat([edge_distances.unsqueeze(-1), edge_type_encoding], dim=1)  # [E, 7] (1 distance + 6 edge types)
        else:
            edge_attr = torch.empty(0, 7, device=device)
        
        # Apply EGNN layers with coordinate updates
        h = combined_features
        x = combined_coords
        
        for i, (egnn_layer, coord_layer) in enumerate(zip(self.egnn_layers, self.coord_update_layers)):
            if combined_edge_index.size(1) > 0:
                # Update node features and coordinates with edge attributes
                h_new, x_new = egnn_layer(h, x, combined_edge_index, edge_attr)
                
                # Additional coordinate updates using coordinate-specific layers
                coord_update = coord_layer(h, x, combined_edge_index, edge_attr)
                x = x_new + coord_update  # Combine both coordinate updates
                h = h_new
        
        # Extract updated coordinates
        current_offset = 0
        
        # Ligand coordinates
        if ligand_coords.size(0) > 0:
            updated_ligand_coords = x[current_offset:current_offset + ligand_coords.size(0)]
            current_offset += ligand_coords.size(0)
        else:
            updated_ligand_coords = torch.empty(0, 3, device=device)
        
        # Skip backbone coordinates (we don't predict them)
        if backbone_coords.size(0) > 0:
            current_offset += backbone_coords.size(0)
        
        # Sidechain coordinates
        if sidechain_coords.size(0) > 0:
            updated_sidechain_coords = x[current_offset:current_offset + sidechain_coords.size(0)]
        else:
            updated_sidechain_coords = torch.empty(0, 3, device=device)
        
        # ===== ORGANIZE OUTPUTS =====
        
        # Organize ligand coordinates by batch
        pred_ligand_coords = torch.zeros(lig_batch_size, max_ligand_atoms, 3, device=device)
        if updated_ligand_coords.size(0) > 0:
            coord_idx = 0
            for b in range(lig_batch_size):
                num_atoms = (ligand_batch_idx == b).sum().item()
                if num_atoms > 0:
                    pred_ligand_coords[b, :num_atoms] = updated_ligand_coords[coord_idx:coord_idx + num_atoms]
                    coord_idx += num_atoms
        
        # Organize sidechain coordinates by batch and residue
        pred_sidechain_coords = torch.zeros(lig_batch_size, num_residues, max_sidechain_atoms, 3, device=device)
        if updated_sidechain_coords.size(0) > 0:
            for i, (batch_id, residue_idx) in enumerate(zip(sidechain_batch_idx, sidechain_residue_idx)):
                # Map global residue index to batch-local residue index
                batch_protein_mask = protein_batch_idx == batch_id
                batch_residue_indices = torch.where(batch_protein_mask)[0]
                
                # Find local residue index within the batch
                local_res_idx = (batch_residue_indices == residue_idx).nonzero()
                if local_res_idx.size(0) > 0:
                    local_res_idx = local_res_idx[0, 0].item()
                    
                    # Find which sidechain atom this is (approximate based on order)
                    atom_idx = 0  # Simplified: assumes sequential ordering
                    if local_res_idx < num_residues and atom_idx < max_sidechain_atoms:
                        pred_sidechain_coords[batch_id, local_res_idx, atom_idx] = updated_sidechain_coords[i]
        
        # Apply masks
        pred_sidechain_coords = pred_sidechain_coords * protein_mask.unsqueeze(-1).unsqueeze(-1)
        pred_sidechain_coords = pred_sidechain_coords * X_sidechain_mask.unsqueeze(-1)
        
        return pred_ligand_coords, pred_sidechain_coords