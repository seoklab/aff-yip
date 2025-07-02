# src/model/my_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPCoordinatePredictor_SidechainMap(nn.Module):
    """
    MLP-based coordinate predictor using sidechain_map directly.
    """

    def __init__(self,
                 lig_embed_dim=196,
                 prot_embed_dim=196,
                 hidden_dim=128,
                 num_layers=3,
                 dropout=0.1,
                 pocket_radius=8.0):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pocket_radius = pocket_radius

        # Feature projection layers
        self.ligand_proj = nn.Linear(lig_embed_dim, hidden_dim)
        self.protein_proj = nn.Linear(prot_embed_dim, hidden_dim)
        self.sidechain_proj = nn.Linear(64, hidden_dim)  # For sidechain atom features

        # MLP layers for processing features
        self.feature_mlps = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.feature_mlps.append(mlp)

        # Coordinate prediction MLPs
        self.ligand_coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # Output 3D coordinates
        )

        self.sidechain_coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # Output 3D coordinates
        )

        # Context MLPs for incorporating surrounding information
        self.ligand_context_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # ligand + protein + sidechain context
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.sidechain_context_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # sidechain + protein + ligand context
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Cross-interaction MLPs for ligand-sidechain communication
        self.ligand_sidechain_interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # ligand + sidechain features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.sidechain_ligand_interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # sidechain + ligand features  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Sidechain atom type embeddings
        self.sidechain_atom_types = ['C', 'N', 'O', 'S', 'P', 'UNK']
        self.sidechain_atom_embedding = nn.Embedding(len(self.sidechain_atom_types), 32)

    def _compute_batch_context(self, features, batch_idx):
        """
        Compute per-batch context by averaging features within each batch.
        """
        if features.size(0) == 0:
            return torch.empty(0, self.hidden_dim, device=features.device)
        batch_context = []
        for batch_id in torch.unique(batch_idx):
            batch_mask = batch_idx == batch_id
            batch_features = features[batch_mask]   
            # Average features within the batch
            context = batch_features.mean(dim=0)
            batch_context.extend([context] * batch_mask.sum().item())
        
        return torch.stack(batch_context)

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

        # Prepare data structures
        atom_features = []
        atom_coords = []
        atom_batch_idx = []
        atom_to_residue_mapping = []  # (batch_id, residue_key, atom_name) for each atom

        # Prepare target and prediction structures
        sidechain_targets = {}
        sidechain_predictions = {}

        protein_res_idx = 0

        for batch_id in range(len(sidechain_maps_list)):
            # Find protein residues in this batch
            batch_protein_mask = protein_batch_idx == batch_id
            num_residues_in_batch = batch_protein_mask.sum().item()

            if num_residues_in_batch == 0:
                protein_res_idx += num_residues_in_batch
                continue

            batch_sidechain_map = sidechain_maps_list[batch_id]
            residue_keys = list(batch_sidechain_map.keys())

            # Get backbone coordinates for this batch
            batch_backbone_coords = backbone_coords[batch_protein_mask]

            # Initialize batch structures
            sidechain_targets[batch_id] = {}
            sidechain_predictions[batch_id] = {}

            # Process residues
            max_residues_to_process = min(num_residues_in_batch, len(residue_keys))

            for res_idx in range(max_residues_to_process):
                residue_key = residue_keys[res_idx]

                if residue_key not in batch_sidechain_map:
                    continue

                sidechain_atoms = batch_sidechain_map[residue_key]

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

    def _compute_cross_interactions(self, ligand_features, ligand_batch_idx, 
                                   sidechain_features, sidechain_batch_idx,
                                   distance_threshold=8.0):
        """
        Compute cross-interactions between ligand and sidechain atoms within distance threshold.
        Returns interaction-aware features for both ligand and sidechain atoms.
        """
        device = ligand_features.device
        
        if ligand_features.size(0) == 0 or sidechain_features.size(0) == 0:
            return ligand_features, sidechain_features
        
        # For each batch, compute ligand-sidechain interactions
        enhanced_ligand_features = []
        enhanced_sidechain_features = []
        
        # Process ligand atoms
        for i, batch_id in enumerate(ligand_batch_idx):
            ligand_feat = ligand_features[i]
            
            # Find sidechain atoms in the same batch
            batch_sidechain_mask = sidechain_batch_idx == batch_id
            if batch_sidechain_mask.any():
                batch_sidechain_features = sidechain_features[batch_sidechain_mask]
                # Average sidechain features as interaction context
                sidechain_context = batch_sidechain_features.mean(dim=0)
                
                # Compute interaction
                combined_feat = torch.cat([ligand_feat, sidechain_context])
                interaction_feat = self.ligand_sidechain_interaction(combined_feat)
                enhanced_ligand_features.append(ligand_feat + interaction_feat)  # Residual
            else:
                enhanced_ligand_features.append(ligand_feat)
        
        # Process sidechain atoms  
        for i, batch_id in enumerate(sidechain_batch_idx):
            sidechain_feat = sidechain_features[i]
            
            # Find ligand atoms in the same batch
            batch_ligand_mask = ligand_batch_idx == batch_id
            if batch_ligand_mask.any():
                batch_ligand_features = ligand_features[batch_ligand_mask]
                # Average ligand features as interaction context
                ligand_context = batch_ligand_features.mean(dim=0)
                
                # Compute interaction
                combined_feat = torch.cat([sidechain_feat, ligand_context])
                interaction_feat = self.sidechain_ligand_interaction(combined_feat)
                enhanced_sidechain_features.append(sidechain_feat + interaction_feat)  # Residual
            else:
                enhanced_sidechain_features.append(sidechain_feat)
        
        if enhanced_ligand_features:
            enhanced_ligand_features = torch.stack(enhanced_ligand_features)
        else:
            enhanced_ligand_features = ligand_features
            
        if enhanced_sidechain_features:
            enhanced_sidechain_features = torch.stack(enhanced_sidechain_features)
        else:
            enhanced_sidechain_features = sidechain_features
            
        return enhanced_ligand_features, enhanced_sidechain_features


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
            # Use initial coordinates (pos) for input, NOT target_pos
            ligand_coords = ligand_batch.pos
            # Store targets separately for later use
            ligand_targets = ligand_batch.target_pos if hasattr(ligand_batch, 'target_pos') else ligand_batch.pos
        else:
            # Fallback: random initialization
            ligand_coords = torch.randn(ligand_embeddings.size(0), 3, device=device) * 2.0
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

        # Apply MLP layers to process features independently
        for mlp in self.feature_mlps:
            if h_ligand.size(0) > 0:
                h_ligand = h_ligand + mlp(h_ligand)  # Residual connection
            if h_protein.size(0) > 0:
                h_protein = h_protein + mlp(h_protein)  # Residual connection
            if h_sidechain.size(0) > 0:
                h_sidechain = h_sidechain + mlp(h_sidechain)  # Residual connection

        # Compute cross-interactions between ligand and sidechain
        if h_ligand.size(0) > 0 and h_sidechain.size(0) > 0:
            h_ligand, h_sidechain = self._compute_cross_interactions(
                h_ligand, ligand_batch_idx, h_sidechain, sidechain_data['batch_idx']
            )

        # Compute context for interaction modeling
        if h_protein.size(0) > 0:
            protein_context_per_batch = self._compute_batch_context(h_protein, protein_batch_idx)
        else:
            protein_context_per_batch = torch.empty(0, self.hidden_dim, device=device)
            
        # Compute ligand and sidechain contexts for cross-talk
        if h_ligand.size(0) > 0:
            ligand_context_per_batch = self._compute_batch_context(h_ligand, ligand_batch_idx)
        else:
            ligand_context_per_batch = torch.empty(0, self.hidden_dim, device=device)
            
        if h_sidechain.size(0) > 0:
            sidechain_context_per_batch = self._compute_batch_context(h_sidechain, sidechain_data['batch_idx'])
        else:
            sidechain_context_per_batch = torch.empty(0, self.hidden_dim, device=device)

        # Process ligand coordinates with context
        ligand_predictions = {}
        if h_ligand.size(0) > 0:
            # Add multi-component context to ligand features
            ligand_context = []
            sidechain_context = []
            for batch_id in ligand_batch_idx:
                # Get protein context
                protein_batch_mask = protein_batch_idx == batch_id
                if protein_batch_mask.any() and protein_context_per_batch.size(0) > 0:
                    batch_protein_context = protein_context_per_batch[protein_batch_mask][0]
                else:
                    batch_protein_context = torch.zeros(self.hidden_dim, device=device)
                ligand_context.append(batch_protein_context)
                
                # Get sidechain context
                sidechain_batch_mask = sidechain_data['batch_idx'] == batch_id
                if sidechain_batch_mask.any() and sidechain_context_per_batch.size(0) > 0:
                    batch_sidechain_context = sidechain_context_per_batch[sidechain_batch_mask][0]
                else:
                    batch_sidechain_context = torch.zeros(self.hidden_dim, device=device)
                sidechain_context.append(batch_sidechain_context)
            
            ligand_context = torch.stack(ligand_context)
            sidechain_context = torch.stack(sidechain_context)
            
            # Combine ligand features with protein + sidechain context
            ligand_with_context = torch.cat([h_ligand, ligand_context, sidechain_context], dim=1)
            ligand_contextual_features = self.ligand_context_mlp(ligand_with_context)
            
            # Predict coordinates
            ligand_coord_deltas = self.ligand_coord_mlp(ligand_contextual_features)
            predicted_ligand_coords = ligand_coords + ligand_coord_deltas
            
            # Organize by batch
            coord_idx = 0
            target_idx = 0
            for batch_id in torch.unique(ligand_batch_idx):
                batch_mask = ligand_batch_idx == batch_id
                num_atoms = batch_mask.sum().item()
                if num_atoms > 0:
                    batch_pred_coords = predicted_ligand_coords[coord_idx:coord_idx + num_atoms]
                    batch_target_coords = ligand_targets[target_idx:target_idx + num_atoms]

                    ligand_predictions[batch_id.item()] = {
                        'predictions': batch_pred_coords,
                        'targets': batch_target_coords
                    }
                    coord_idx += num_atoms
                    target_idx += num_atoms

        # Process sidechain coordinates with context
        sidechain_predictions = sidechain_data['sidechain_predictions'].copy()
        
        if h_sidechain.size(0) > 0:
            # Add multi-component context to sidechain features
            sidechain_protein_context = []
            sidechain_ligand_context = []
            for batch_id in sidechain_data['batch_idx']:
                # Get protein context
                protein_batch_mask = protein_batch_idx == batch_id
                if protein_batch_mask.any() and protein_context_per_batch.size(0) > 0:
                    batch_protein_context = protein_context_per_batch[protein_batch_mask][0]
                else:
                    batch_protein_context = torch.zeros(self.hidden_dim, device=device)
                sidechain_protein_context.append(batch_protein_context)
                
                # Get ligand context
                ligand_batch_mask = ligand_batch_idx == batch_id
                if ligand_batch_mask.any() and ligand_context_per_batch.size(0) > 0:
                    batch_ligand_context = ligand_context_per_batch[ligand_batch_mask][0]
                else:
                    batch_ligand_context = torch.zeros(self.hidden_dim, device=device)
                sidechain_ligand_context.append(batch_ligand_context)
            
            sidechain_protein_context = torch.stack(sidechain_protein_context)
            sidechain_ligand_context = torch.stack(sidechain_ligand_context)
            
            # Combine sidechain features with protein + ligand context
            sidechain_with_context = torch.cat([h_sidechain, sidechain_protein_context, sidechain_ligand_context], dim=1)
            sidechain_contextual_features = self.sidechain_context_mlp(sidechain_with_context)
            
            # Predict coordinates
            sidechain_coord_deltas = self.sidechain_coord_mlp(sidechain_contextual_features)
            predicted_sidechain_coords = sidechain_data['coords'] + sidechain_coord_deltas
            
            # Organize sidechain predictions using the atom mapping
            for i, (batch_id, residue_key, atom_name) in enumerate(sidechain_data['atom_to_residue_mapping']):
                if i < predicted_sidechain_coords.size(0):
                    sidechain_predictions[batch_id][residue_key][atom_name] = predicted_sidechain_coords[i]

        # Prepare split features for compatibility (simplified for MLP)
        split_features = {
            'ligand_features': {},
            'protein_features': {},
            'sidechain_features': {}
        }
        
        # Organize features by batch (optional, for compatibility)
        if h_ligand.size(0) > 0:
            feature_idx = 0
            for batch_id in torch.unique(ligand_batch_idx):
                batch_mask = ligand_batch_idx == batch_id
                num_nodes = batch_mask.sum().item()
                split_features['ligand_features'][batch_id.item()] = h_ligand[feature_idx:feature_idx + num_nodes]
                feature_idx += num_nodes
        
        if h_protein.size(0) > 0:
            feature_idx = 0
            for batch_id in torch.unique(protein_batch_idx):
                batch_mask = protein_batch_idx == batch_id
                num_nodes = batch_mask.sum().item()
                split_features['protein_features'][batch_id.item()] = h_protein[feature_idx:feature_idx + num_nodes]
                feature_idx += num_nodes
        
        if h_sidechain.size(0) > 0:
            feature_idx = 0
            for batch_id in torch.unique(sidechain_data['batch_idx']):
                batch_mask = sidechain_data['batch_idx'] == batch_id
                num_nodes = batch_mask.sum().item()
                split_features['sidechain_features'][batch_id.item()] = h_sidechain[feature_idx:feature_idx + num_nodes]
                feature_idx += num_nodes

        # Prepare final result
        result = {
            'ligand_coords': ligand_predictions,
            'sidechain_predictions': sidechain_predictions,
            'sidechain_targets': sidechain_data['sidechain_targets']
        }

        return result, split_features

# Modified MLP Coordinate Predictor using Sidechain Map approach
class MLPCoordinatePredictor_SidechainMap_no_virtual_ligand_interaction(nn.Module):
    """
    MLP-based coordinate predictor using sidechain_map directly.
    """

    def __init__(self,
                 lig_embed_dim=196,
                 prot_embed_dim=196,
                 hidden_dim=128,
                 num_layers=3,
                 dropout=0.1,
                 pocket_radius=8.0):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pocket_radius = pocket_radius

        # Feature projection layers
        self.ligand_proj = nn.Linear(lig_embed_dim, hidden_dim)
        self.protein_proj = nn.Linear(prot_embed_dim, hidden_dim)
        self.sidechain_proj = nn.Linear(64, hidden_dim)  # For sidechain atom features

        # MLP layers for processing features
        self.feature_mlps = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.feature_mlps.append(mlp)

        # Coordinate prediction MLPs
        self.ligand_coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # Output 3D coordinates
        )

        self.sidechain_coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # Output 3D coordinates
        )

        # Context MLPs for incorporating surrounding information
        self.ligand_context_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # ligand + protein context
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.sidechain_context_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # sidechain + protein context
            nn.ReLU(),
            nn.Dropout(dropout)
        )

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

        # Prepare data structures
        atom_features = []
        atom_coords = []
        atom_batch_idx = []
        atom_to_residue_mapping = []  # (batch_id, residue_key, atom_name) for each atom

        # Prepare target and prediction structures
        sidechain_targets = {}
        sidechain_predictions = {}

        protein_res_idx = 0

        for batch_id in range(len(sidechain_maps_list)):
            # Find protein residues in this batch
            batch_protein_mask = protein_batch_idx == batch_id
            num_residues_in_batch = batch_protein_mask.sum().item()

            if num_residues_in_batch == 0:
                protein_res_idx += num_residues_in_batch
                continue

            batch_sidechain_map = sidechain_maps_list[batch_id]
            residue_keys = list(batch_sidechain_map.keys())

            # Get backbone coordinates for this batch
            batch_backbone_coords = backbone_coords[batch_protein_mask]

            # Initialize batch structures
            sidechain_targets[batch_id] = {}
            sidechain_predictions[batch_id] = {}

            # Process residues
            max_residues_to_process = min(num_residues_in_batch, len(residue_keys))

            for res_idx in range(max_residues_to_process):
                residue_key = residue_keys[res_idx]

                if residue_key not in batch_sidechain_map:
                    continue

                sidechain_atoms = batch_sidechain_map[residue_key]

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

    def _compute_batch_context(self, features, batch_idx):
        """
        Compute per-batch context by averaging features within each batch.
        """
        if features.size(0) == 0:
            return torch.empty(0, self.hidden_dim, device=features.device)
        
        batch_context = []
        for batch_id in torch.unique(batch_idx):
            batch_mask = batch_idx == batch_id
            batch_features = features[batch_mask]
            # Average features within the batch
            context = batch_features.mean(dim=0)
            batch_context.extend([context] * batch_mask.sum().item())
        
        return torch.stack(batch_context)

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
            # Use initial coordinates (pos) for input, NOT target_pos
            ligand_coords = ligand_batch.pos
            # Store targets separately for later use
            ligand_targets = ligand_batch.target_pos if hasattr(ligand_batch, 'target_pos') else ligand_batch.pos
        else:
            # Fallback: random initialization
            ligand_coords = torch.randn(ligand_embeddings.size(0), 3, device=device) * 2.0
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

        # Apply MLP layers to process features independently
        for mlp in self.feature_mlps:
            if h_ligand.size(0) > 0:
                h_ligand = h_ligand + mlp(h_ligand)  # Residual connection
            if h_protein.size(0) > 0:
                h_protein = h_protein + mlp(h_protein)  # Residual connection
            if h_sidechain.size(0) > 0:
                h_sidechain = h_sidechain + mlp(h_sidechain)  # Residual connection

        # Compute context for interaction modeling
        if h_protein.size(0) > 0:
            protein_context_per_batch = self._compute_batch_context(h_protein, protein_batch_idx)
        else:
            protein_context_per_batch = torch.empty(0, self.hidden_dim, device=device)

        # Process ligand coordinates with context
        ligand_predictions = {}
        if h_ligand.size(0) > 0 and protein_context_per_batch.size(0) > 0:
            # Add context to ligand features
            ligand_context = []
            for batch_id in ligand_batch_idx:
                # Find corresponding protein context for this batch
                protein_batch_mask = protein_batch_idx == batch_id
                if protein_batch_mask.any():
                    batch_protein_context = protein_context_per_batch[protein_batch_mask][0]  # Take first (they're all the same for the batch)
                    ligand_context.append(batch_protein_context)
                else:
                    ligand_context.append(torch.zeros(self.hidden_dim, device=device))
            
            ligand_context = torch.stack(ligand_context)
            
            # Combine ligand features with protein context
            ligand_with_context = torch.cat([h_ligand, ligand_context], dim=1)
            ligand_contextual_features = self.ligand_context_mlp(ligand_with_context)
            
            # Predict coordinates
            ligand_coord_deltas = self.ligand_coord_mlp(ligand_contextual_features)
            predicted_ligand_coords = ligand_coords + ligand_coord_deltas
            
            # Organize by batch
            coord_idx = 0
            target_idx = 0
            for batch_id in torch.unique(ligand_batch_idx):
                batch_mask = ligand_batch_idx == batch_id
                num_atoms = batch_mask.sum().item()
                if num_atoms > 0:
                    batch_pred_coords = predicted_ligand_coords[coord_idx:coord_idx + num_atoms]
                    batch_target_coords = ligand_targets[target_idx:target_idx + num_atoms]

                    ligand_predictions[batch_id.item()] = {
                        'predictions': batch_pred_coords,
                        'targets': batch_target_coords
                    }
                    coord_idx += num_atoms
                    target_idx += num_atoms

        # Process sidechain coordinates with context
        sidechain_predictions = sidechain_data['sidechain_predictions'].copy()
        
        if h_sidechain.size(0) > 0 and protein_context_per_batch.size(0) > 0:
            # Add context to sidechain features
            sidechain_context = []
            for batch_id in sidechain_data['batch_idx']:
                # Find corresponding protein context for this batch
                protein_batch_mask = protein_batch_idx == batch_id
                if protein_batch_mask.any():
                    batch_protein_context = protein_context_per_batch[protein_batch_mask][0]
                    sidechain_context.append(batch_protein_context)
                else:
                    sidechain_context.append(torch.zeros(self.hidden_dim, device=device))
            
            sidechain_context = torch.stack(sidechain_context)
            
            # Combine sidechain features with protein context
            sidechain_with_context = torch.cat([h_sidechain, sidechain_context], dim=1)
            sidechain_contextual_features = self.sidechain_context_mlp(sidechain_with_context)
            
            # Predict coordinates
            sidechain_coord_deltas = self.sidechain_coord_mlp(sidechain_contextual_features)
            predicted_sidechain_coords = sidechain_data['coords'] + sidechain_coord_deltas
            
            # Organize sidechain predictions using the atom mapping
            for i, (batch_id, residue_key, atom_name) in enumerate(sidechain_data['atom_to_residue_mapping']):
                if i < predicted_sidechain_coords.size(0):
                    sidechain_predictions[batch_id][residue_key][atom_name] = predicted_sidechain_coords[i]

        # Prepare split features for compatibility (simplified for MLP)
        split_features = {
            'ligand_features': {},
            'protein_features': {},
            'sidechain_features': {}
        }
        
        # Organize features by batch (optional, for compatibility)
        if h_ligand.size(0) > 0:
            feature_idx = 0
            for batch_id in torch.unique(ligand_batch_idx):
                batch_mask = ligand_batch_idx == batch_id
                num_nodes = batch_mask.sum().item()
                split_features['ligand_features'][batch_id.item()] = h_ligand[feature_idx:feature_idx + num_nodes]
                feature_idx += num_nodes
        
        if h_protein.size(0) > 0:
            feature_idx = 0
            for batch_id in torch.unique(protein_batch_idx):
                batch_mask = protein_batch_idx == batch_id
                num_nodes = batch_mask.sum().item()
                split_features['protein_features'][batch_id.item()] = h_protein[feature_idx:feature_idx + num_nodes]
                feature_idx += num_nodes
        
        if h_sidechain.size(0) > 0:
            feature_idx = 0
            for batch_id in torch.unique(sidechain_data['batch_idx']):
                batch_mask = sidechain_data['batch_idx'] == batch_id
                num_nodes = batch_mask.sum().item()
                split_features['sidechain_features'][batch_id.item()] = h_sidechain[feature_idx:feature_idx + num_nodes]
                feature_idx += num_nodes

        # Prepare final result
        result = {
            'ligand_coords': ligand_predictions,
            'sidechain_predictions': sidechain_predictions,
            'sidechain_targets': sidechain_data['sidechain_targets']
        }

        return result, split_features