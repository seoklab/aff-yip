# Modified EGNN Coordinate Predictor using Sidechain Map approach
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean, scatter_add
from .egnn_layers import EGNNConv, CoordEGNNConv, CoordEGNNConv_v2, CoordEGNNConv_v3
import numpy as np

class EGNNCoordinatePredictor_SidechainMap(nn.Module):
    """
    EGNN-based coordinate predictor using sidechain_map directly.
    No padding, no masking - predict exactly what exists!
    """

    def __init__(self,
                 lig_embed_dim=196,
                 prot_embed_dim=196,
                 hidden_dim=128,
                 num_layers=3,
                 dropout=0.1,
                 coord_update_type="v2",
                 pocket_radius=8.0):
        super().__init__()

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

        # Coordinate update layers
        self.coord_update_layers = nn.ModuleList()
        for _ in range(num_layers):
            if coord_update_type == "v2":
                coord_layer = CoordEGNNConv_v2(
                    in_size=hidden_dim,
                    hidden_size=hidden_dim,
                    dropout=dropout,
                    edge_feat_size=7
                )
            elif coord_update_type == "v3":
                coord_layer = CoordEGNNConv_v3(
                    in_size=hidden_dim,
                    hidden_size=hidden_dim,
                    dropout=dropout,
                    edge_feat_size=7
                )
            else:  # v1 or default
                coord_layer = CoordEGNNConv(
                    in_size=hidden_dim,
                    hidden_size=hidden_dim,
                    dropout=dropout,
                    edge_feat_size=7
                )
            self.coord_update_layers.append(coord_layer)

        # Node type embeddings
        self.node_type_embedding = nn.Embedding(4, hidden_dim)  # 0:ligand, 1:backbone, 2:sidechain, 3:virtual

        # Sidechain atom type embeddings
        self.sidechain_atom_types = ['C', 'N', 'O', 'S', 'P', 'UNK']
        self.sidechain_atom_embedding = nn.Embedding(len(self.sidechain_atom_types), 32)

    def _create_sidechain_nodes_from_map(self, protein_embeddings, protein_batch_idx,
                                       backbone_coords, sidechain_maps):
        """
        Create sidechain atom nodes directly from sidechain_map.
        Returns organized data structures for easy loss calculation.
        """
        device = protein_embeddings.device

        # Handle sidechain_maps - could be tuple/list or single dict
        if isinstance(sidechain_maps, (tuple, list)):
            sidechain_maps_list = sidechain_maps
        else:
            sidechain_maps_list = [sidechain_maps]

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

        # Prepare data structures - we'll build these in order
        atom_features = []
        atom_coords = []
        atom_batch_idx = []
        atom_to_residue_mapping = []  # (batch_id, residue_key, atom_name) for each atom

        # Prepare target and prediction structures
        sidechain_targets = {}
        sidechain_predictions = {}

        # print(f"DEBUG EGNN: Processing {len(sidechain_maps_list)} batches")

        protein_res_idx = 0

        for batch_id in range(len(sidechain_maps_list)):
            # Find protein residues in this batch
            batch_protein_mask = protein_batch_idx == batch_id
            num_residues_in_batch = batch_protein_mask.sum().item()

            # print(f"DEBUG EGNN: Batch {batch_id}: {num_residues_in_batch} residues in protein_batch_idx")

            if num_residues_in_batch == 0:
                protein_res_idx += num_residues_in_batch
                continue

            batch_sidechain_map = sidechain_maps_list[batch_id]
            residue_keys = list(batch_sidechain_map.keys())

            # print(f"DEBUG EGNN: Batch {batch_id}: {len(residue_keys)} residues in sidechain_map")

            # Get backbone coordinates for this batch
            batch_backbone_coords = backbone_coords[batch_protein_mask]

            # Initialize batch structures
            sidechain_targets[batch_id] = {}
            sidechain_predictions[batch_id] = {}

            # Process residues (only up to the minimum of available data)
            max_residues_to_process = min(num_residues_in_batch, len(residue_keys))
            # print(f"DEBUG EGNN: Will process {max_residues_to_process} residues")

            for res_idx in range(max_residues_to_process):
                residue_key = residue_keys[res_idx]

                # print(f"DEBUG EGNN: Processing residue {res_idx}/{max_residues_to_process}: {residue_key}")

                if residue_key not in batch_sidechain_map:
                    continue

                sidechain_atoms = batch_sidechain_map[residue_key]
                # print(f"DEBUG EGNN: Residue {residue_key} has {len(sidechain_atoms)} atoms: {list(sidechain_atoms.keys())}")

                # Initialize prediction and target structures for this residue
                sidechain_predictions[batch_id][residue_key] = {}
                sidechain_targets[batch_id][residue_key] = {}

                # Create nodes for each sidechain atom
                for atom_name, atom_coords_np in sidechain_atoms.items():
                    # Store target coordinates
                    target_coord = torch.tensor(atom_coords_np, dtype=torch.float32, device=device)
                    sidechain_targets[batch_id][residue_key][atom_name] = target_coord

                    # Create atom features
                    atom_type_idx = get_atom_type_idx(atom_name)
                    atom_type_emb = self.sidechain_atom_embedding(
                        torch.tensor(atom_type_idx, device=device)
                    )

                    # Use residue embedding as base feature
                    residue_emb = protein_embeddings[protein_res_idx + res_idx]

                    # Combine features (ensure total is 64 for sidechain_proj)
                    atom_features_combined = torch.cat([residue_emb[:32], atom_type_emb], dim=0)
                    atom_features.append(atom_features_combined)

                    # Random initial position near CA (don't use ground truth)
                    ca_coord = batch_backbone_coords[res_idx]
                    random_offset = torch.randn(3, device=device) * 1.0
                    random_offset = random_offset / (random_offset.norm() + 1e-8) * (1.0 + torch.rand(1, device=device) * 3.0)
                    initial_coord = ca_coord + random_offset
                    atom_coords.append(initial_coord)

                    atom_batch_idx.append(batch_id)

                    # Store the mapping for this atom
                    atom_to_residue_mapping.append((batch_id, residue_key, atom_name))

            protein_res_idx += num_residues_in_batch

        # print(f"DEBUG EGNN: Created {len(atom_features)} sidechain nodes")

        # Convert to tensors
        if atom_features:
            atom_features_tensor = torch.stack(atom_features)
            atom_coords_tensor = torch.stack(atom_coords)
            atom_batch_idx_tensor = torch.tensor(atom_batch_idx, device=device)
        else:
            atom_features_tensor = torch.empty(0, 64, device=device)
            atom_coords_tensor = torch.empty(0, 3, device=device)
            atom_batch_idx_tensor = torch.empty(0, dtype=torch.long, device=device)

        return {
            'features': atom_features_tensor,
            'coords': atom_coords_tensor,
            'batch_idx': atom_batch_idx_tensor,
            'atom_to_residue_mapping': atom_to_residue_mapping,
            'sidechain_targets': sidechain_targets,
            'sidechain_predictions': sidechain_predictions
        }

    def _create_node_mapping(self, ligand_batch_idx, protein_batch_idx, sidechain_batch_idx):
        """
        Create mapping to track which nodes belong to which component and batch.
        Returns offset information for splitting combined features back.
        """

        # Create mapping structure
        node_mapping = {
            'ligand': {'start': 0, 'end': 0, 'batch_sizes': {}},
            'protein': {'start': 0, 'end': 0, 'batch_sizes': {}},
            'sidechain': {'start': 0, 'end': 0, 'batch_sizes': {}}
        }

        current_offset = 0

        # Track ligand nodes
        if ligand_batch_idx.size(0) > 0:
            node_mapping['ligand']['start'] = current_offset
            node_mapping['ligand']['end'] = current_offset + ligand_batch_idx.size(0)

            # Count nodes per batch for ligand
            for batch_id in torch.unique(ligand_batch_idx):
                batch_mask = ligand_batch_idx == batch_id
                node_mapping['ligand']['batch_sizes'][batch_id.item()] = batch_mask.sum().item()

            current_offset += ligand_batch_idx.size(0)

        # Track protein nodes
        if protein_batch_idx.size(0) > 0:
            node_mapping['protein']['start'] = current_offset
            node_mapping['protein']['end'] = current_offset + protein_batch_idx.size(0)

            # Count nodes per batch for protein
            for batch_id in torch.unique(protein_batch_idx):
                batch_mask = protein_batch_idx == batch_id
                node_mapping['protein']['batch_sizes'][batch_id.item()] = batch_mask.sum().item()

            current_offset += protein_batch_idx.size(0)

        # Track sidechain nodes
        if sidechain_batch_idx.size(0) > 0:
            node_mapping['sidechain']['start'] = current_offset
            node_mapping['sidechain']['end'] = current_offset + sidechain_batch_idx.size(0)

            # Count nodes per batch for sidechain
            for batch_id in torch.unique(sidechain_batch_idx):
                batch_mask = sidechain_batch_idx == batch_id
                node_mapping['sidechain']['batch_sizes'][batch_id.item()] = batch_mask.sum().item()

        return node_mapping

    def _build_combined_graph(self, ligand_batch_idx, ligand_coords, ligand_edge_index,
                             protein_batch_idx, backbone_coords,
                             sidechain_batch_idx, sidechain_coords):
        """Build combined graph with proper edge formation"""
        device = ligand_coords.device
        all_coords = []
        all_batch_idx = []
        node_types = []

        current_offset = 0

        # Add ligand nodes (type 0)
        if ligand_coords.size(0) > 0:
            all_coords.append(ligand_coords)
            all_batch_idx.append(ligand_batch_idx)
            node_types.append(torch.zeros(ligand_coords.size(0), device=device, dtype=torch.long))
            current_offset += ligand_coords.size(0)

        # Add backbone nodes (type 1)
        if backbone_coords.size(0) > 0:
            all_coords.append(backbone_coords)
            all_batch_idx.append(protein_batch_idx)
            node_types.append(torch.ones(backbone_coords.size(0), device=device, dtype=torch.long))
            current_offset += backbone_coords.size(0)

        # Add sidechain nodes (type 2)
        if sidechain_coords.size(0) > 0:
            all_coords.append(sidechain_coords)
            all_batch_idx.append(sidechain_batch_idx)
            node_types.append(2 * torch.ones(sidechain_coords.size(0), device=device, dtype=torch.long))

        if not all_coords:
            return (torch.empty(0, 3, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(2, 0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, device=device),
                    torch.empty(0, 6, device=device))

        # Concatenate all coordinates and batch indices
        combined_coords = torch.cat(all_coords, dim=0)
        combined_batch_idx = torch.cat(all_batch_idx, dim=0)
        combined_node_types = torch.cat(node_types, dim=0)

        # Build edges (simplified for now)
        edge_sources = []
        edge_targets = []
        edge_distances = []
        edge_type_encoding = []

        # Simple approach: connect nearby nodes within each batch
        for batch_id in torch.unique(combined_batch_idx):
            batch_mask = combined_batch_idx == batch_id
            batch_coords = combined_coords[batch_mask]
            batch_indices = torch.where(batch_mask)[0]

            if batch_coords.size(0) < 2:
                continue

            # Compute distance matrix
            dist_matrix = torch.cdist(batch_coords, batch_coords)

            # Connect nodes within reasonable distance
            close_pairs = torch.where((dist_matrix < 8.0) & (dist_matrix > 0))

            for i, j in zip(close_pairs[0], close_pairs[1]):
                global_i = batch_indices[i]
                global_j = batch_indices[j]

                edge_sources.extend([global_i, global_j])
                edge_targets.extend([global_j, global_i])

                dist = dist_matrix[i, j]
                edge_distances.extend([dist, dist])

                # Simple edge type encoding
                edge_type = [1, 0, 0, 0, 0, 0]  # Default type
                edge_type_encoding.extend([edge_type, edge_type])

        if edge_sources:
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long, device=device)
            edge_distances = torch.stack(edge_distances)
            edge_type_encoding = torch.tensor(edge_type_encoding, dtype=torch.float, device=device)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            edge_distances = torch.empty(0, device=device)
            edge_type_encoding = torch.empty(0, 6, device=device)

        return (combined_coords, combined_batch_idx, edge_index, combined_node_types,
                edge_distances, edge_type_encoding)
    def _split_combined_features(self, combined_features, combined_batch_idx, node_mapping):
        """
        Split combined features back into components organized by batch.
        """
        result = {
            'ligand_features': {},
            'protein_features': {},
            'sidechain_features': {}
        }

        # Split ligand features by batch
        if node_mapping['ligand']['end'] > node_mapping['ligand']['start']:
            ligand_features = combined_features[node_mapping['ligand']['start']:node_mapping['ligand']['end']]
            ligand_batch_idx = combined_batch_idx[node_mapping['ligand']['start']:node_mapping['ligand']['end']]

            feature_idx = 0
            for batch_id, num_nodes in node_mapping['ligand']['batch_sizes'].items():
                result['ligand_features'][batch_id] = ligand_features[feature_idx:feature_idx + num_nodes]
                feature_idx += num_nodes

        # Split protein features by batch
        if node_mapping['protein']['end'] > node_mapping['protein']['start']:
            protein_features = combined_features[node_mapping['protein']['start']:node_mapping['protein']['end']]
            protein_batch_idx = combined_batch_idx[node_mapping['protein']['start']:node_mapping['protein']['end']]

            feature_idx = 0
            for batch_id, num_nodes in node_mapping['protein']['batch_sizes'].items():
                result['protein_features'][batch_id] = protein_features[feature_idx:feature_idx + num_nodes]
                feature_idx += num_nodes

        # Split sidechain features by batch
        if node_mapping['sidechain']['end'] > node_mapping['sidechain']['start']:
            sidechain_features = combined_features[node_mapping['sidechain']['start']:node_mapping['sidechain']['end']]
            sidechain_batch_idx = combined_batch_idx[node_mapping['sidechain']['start']:node_mapping['sidechain']['end']]

            feature_idx = 0
            for batch_id, num_nodes in node_mapping['sidechain']['batch_sizes'].items():
                result['sidechain_features'][batch_id] = sidechain_features[feature_idx:feature_idx + num_nodes]
                feature_idx += num_nodes

        return result
    def forward(self,
                ligand_embeddings,      # [N_ligand, embed_dim]
                ligand_batch_idx,       # [N_ligand]
                protein_embeddings,     # [N_protein, embed_dim]
                protein_batch_idx,      # [N_protein]
                sidechain_map,          # List/tuple of dicts with actual sidechain data
                protein_virtual_batch,  # Original batch object
                **kwargs):              # Additional inputs

        device = ligand_embeddings.device

        # Get coordinates from the ligand batch
        ligand_batch = kwargs.get('ligand_batch')
        if ligand_batch is not None:
            # Use initial coordinates (pos) for EGNN input, NOT target_pos
            ligand_coords = ligand_batch.pos
            ligand_edge_index = ligand_batch.edge_index

            # Store targets separately for later use
            ligand_targets = ligand_batch.target_pos if hasattr(ligand_batch, 'target_pos') else ligand_batch.pos
        else:
            # Fallback: random initialization
            ligand_coords = torch.randn(ligand_embeddings.size(0), 3, device=device) * 2.0
            ligand_edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            ligand_targets = ligand_coords.clone()

        # Get backbone coordinates
        protein_mask_1d = protein_virtual_batch.node_type == 0
        backbone_coords = protein_virtual_batch.x[protein_mask_1d]

        # Create sidechain nodes using sidechain_map
        sidechain_data = self._create_sidechain_nodes_from_map(
            protein_embeddings, protein_batch_idx, backbone_coords, sidechain_map
        )

        # Project all features to hidden dimension
        h_ligand = self.ligand_proj(ligand_embeddings) if ligand_embeddings.size(0) > 0 else torch.empty(0, self.hidden_dim, device=device)
        h_protein = self.protein_proj(protein_embeddings) if protein_embeddings.size(0) > 0 else torch.empty(0, self.hidden_dim, device=device)
        h_sidechain = self.sidechain_proj(sidechain_data['features']) if sidechain_data['features'].size(0) > 0 else torch.empty(0, self.hidden_dim, device=device)

        # Build combined graph
        combined_coords, combined_batch_idx, combined_edge_index, combined_node_types, edge_distances, edge_type_encoding = \
            self._build_combined_graph(
                ligand_batch_idx, ligand_coords, ligand_edge_index,
                protein_batch_idx, backbone_coords,
                sidechain_data['batch_idx'], sidechain_data['coords']
            )
        # node mapping for splitting features later
        node_mapping = self._create_node_mapping(
            ligand_batch_idx, protein_batch_idx, sidechain_data['batch_idx']
        )
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
            return {
                'ligand_coords': {},
                'sidechain_predictions': sidechain_data['sidechain_predictions'],
                'sidechain_targets': sidechain_data['sidechain_targets']
            }

        combined_features = torch.cat(all_features, dim=0)

        # Add node type embeddings
        if combined_node_types.size(0) > 0:
            type_embeddings = self.node_type_embedding(combined_node_types)
            combined_features = combined_features + type_embeddings

        # Create edge attributes
        if combined_edge_index.size(1) > 0:
            edge_attr = torch.cat([edge_distances.unsqueeze(-1), edge_type_encoding], dim=1)
        else:
            edge_attr = torch.empty(0, 7, device=device)

        # Apply EGNN layers
        h = combined_features
        x = combined_coords

        for egnn_layer, coord_layer in zip(self.egnn_layers, self.coord_update_layers):
            if combined_edge_index.size(1) > 0:
                h_new, x_new = egnn_layer(h, x, combined_edge_index, edge_attr)
                coord_update = coord_layer(h, x, combined_edge_index, edge_attr)
                x = x_new + coord_update
                h = h_new

        split_features = self._split_combined_features(h, combined_batch_idx, node_mapping)

        # Extract predictions and organize by residue
        result = {
            'ligand_coords': {},
            'sidechain_predictions': sidechain_data['sidechain_predictions'].copy(),
            'sidechain_targets': sidechain_data['sidechain_targets']
        }

        # Organize ligand predictions
        current_offset = 0
        if ligand_coords.size(0) > 0:
            updated_ligand_coords = x[current_offset:current_offset + ligand_coords.size(0)]
            current_offset += ligand_coords.size(0)

            # Organize by batch
            coord_idx = 0
            target_idx = 0
            for batch_id in torch.unique(ligand_batch_idx):
                batch_mask = ligand_batch_idx == batch_id
                num_atoms = batch_mask.sum().item()
                if num_atoms > 0:
                    batch_pred_coords = updated_ligand_coords[coord_idx:coord_idx + num_atoms]
                    batch_target_coords = ligand_targets[target_idx:target_idx + num_atoms]

                    result['ligand_coords'][batch_id.item()] = {
                        'predictions': batch_pred_coords,
                        'targets': batch_target_coords
                    }
                    coord_idx += num_atoms
                    target_idx += num_atoms

        # Skip backbone coordinates
        if backbone_coords.size(0) > 0:
            current_offset += backbone_coords.size(0)

        # Organize sidechain predictions using the atom mapping
        if sidechain_data['coords'].size(0) > 0:
            updated_sidechain_coords = x[current_offset:current_offset + sidechain_data['coords'].size(0)]

            # print(f"DEBUG EGNN: Filling {len(sidechain_data['atom_to_residue_mapping'])} sidechain predictions")
            # print(f"DEBUG EGNN: Coordinate tensor size: {updated_sidechain_coords.size(0)}")

            atoms_filled = 0
            for i, (batch_id, residue_key, atom_name) in enumerate(sidechain_data['atom_to_residue_mapping']):
                if i < updated_sidechain_coords.size(0):
                    # Directly assign the prediction using the mapping
                    result['sidechain_predictions'][batch_id][residue_key][atom_name] = updated_sidechain_coords[i]
                    atoms_filled += 1

                    # if i < 5:  # Only print first few for debugging
                    #     print(f"DEBUG EGNN: ✓ Filled {residue_key} {atom_name} (position {i})")
                else:
                    # print(f"DEBUG EGNN: ✗ Index {i} out of bounds for coordinate tensor of size {updated_sidechain_coords.size(0)}")
                    break

            # print(f"DEBUG EGNN: Successfully filled {atoms_filled} atom predictions")

            # Verify the results
            total_predictions = 0
            for batch_id in result['sidechain_predictions']:
                for residue_key in result['sidechain_predictions'][batch_id]:
                    num_atoms = len(result['sidechain_predictions'][batch_id][residue_key])
                    total_predictions += num_atoms

            # print(f"DEBUG EGNN: Total predictions stored: {total_predictions}")
        # for key in split_features:
        #     # print shapes
        #     for idx in split_features[key]:
        #         print(f"DEBUG EGNN: {key} features shape: {split_features[key][idx].shape}")
        return result, split_features
