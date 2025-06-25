# src/data/featurizers/protein.py
import torch
import torch_cluster
import torch.nn.functional as F # For one_hot in edge_type
import numpy as np
from torch_geometric.data import Data

from ..structures import Protein, Ligand, VirtualNode # For type hinting and VirtualNode usage
from .utils import (
    stack_residue_coordinates, get_residue_dihedrals, get_sidechain_orientation,
    get_backbone_orientation, _normalize_torch, get_rbf, get_positional_embeddings,
    stack_water_coordinates, generate_virtual_nodes, get_water_embeddings, get_aa_one_hot
)
from ...io.write_pdb import write_virtual_nodes_pdb # Adjusted import for io

class ProteinFeaturizer:
    def __init__(self, top_k: int = 30):
        self.top_k = top_k

    def featurize_no_water_graph(self, protein: Protein, center=None, crop_size=None) -> Data:
        X_res_all_coords, res_list = stack_residue_coordinates(protein) # All atom coordinates for all residues
        if X_res_all_coords.numel() == 0: # Handle empty protein
             return Data(x=torch.empty(0,3), node_s=torch.empty(0,6), node_v=torch.empty(0,2,3,3), edge_index=torch.empty(2,0), edge_s=torch.empty(0,32), edge_v=torch.empty(0,3))

        X_res_ca = X_res_all_coords[1::3] if X_res_all_coords.size(0) >= 2 else X_res_all_coords # CA atoms, handle case with few atoms
        X_all_coords = X_res_ca # Coordinates for graph construction (CA only)

        if center is not None and crop_size is not None:
            mask_res_ca = ((X_res_ca - center).abs() < crop_size / 2).all(dim=-1)
            X_res_ca = X_res_ca[mask_res_ca]
            X_all_coords = X_res_ca
            # Need to find original indices of kept CAs to slice features derived from X_res_all_coords
            keep_res_idx = mask_res_ca.nonzero(as_tuple=False).view(-1)
        else:
            keep_res_idx = torch.arange(X_res_ca.size(0))

        if X_all_coords.numel() == 0: # Handle case where cropping results in no atoms
            return Data(x=torch.empty(0,3), node_s=torch.empty(0,6), node_v=torch.empty(0,2,3,3), edge_index=torch.empty(2,0), edge_s=torch.empty(0,32), edge_v=torch.empty(0,3))


        node_type = torch.zeros(X_res_ca.size(0)).long() # Only protein residues

        # Features are typically derived from all atoms, then indexed for the CAs
        # Assuming X_res_all_coords corresponds to original unfiltered set of atoms for all residues
        node_s_all_residues = get_residue_dihedrals(X_res_all_coords) # Dihedrals for all residues
        node_s = node_s_all_residues[keep_res_idx] if node_s_all_residues.numel() > 0 else torch.empty((0, node_s_all_residues.size(1) if node_s_all_residues.ndim > 1 else 6 ))


        sidechain_orient_all = get_sidechain_orientation(X_res_all_coords)
        backbone_orient_all = get_backbone_orientation(X_res_all_coords)
        node_v_all_residues = torch.cat([sidechain_orient_all.unsqueeze(-2), backbone_orient_all], dim=-2) if sidechain_orient_all.numel() > 0 and backbone_orient_all.numel() > 0 else torch.empty((0,0,3,3)) # Check dimensions
        node_v = node_v_all_residues[keep_res_idx] if node_v_all_residues.numel() > 0 else torch.empty((0, node_v_all_residues.size(1) if node_v_all_residues.ndim > 1 else 3, 3,3))


        edge_index = torch_cluster.knn_graph(X_all_coords, k=min(self.top_k, X_all_coords.size(0)-1) if X_all_coords.size(0)>1 else 0 )
        E_vectors = X_all_coords[edge_index[0]] - X_all_coords[edge_index[1]] if edge_index.numel() > 0 else torch.empty((0,3))
        edge_v_2D = _normalize_torch(E_vectors) if E_vectors.numel() > 0 else torch.empty((0,3))
        edge_v = edge_v_2D.unsqueeze(1) if edge_v_2D.numel() > 0 else torch.empty((0,1,3))
        edge_dist = E_vectors.norm(dim=-1) if E_vectors.numel() > 0 else torch.empty((0))
        
        rbf_features = get_rbf(edge_dist, D_count=16, device=edge_index.device) if edge_dist.numel() > 0 else torch.empty((0,16))
        pos_emb_features = get_positional_embeddings(edge_index, num_embeddings=16) if edge_index.numel() > 0 else torch.empty((0,16))
        edge_s = torch.cat([rbf_features, pos_emb_features], dim=-1)
        # print data shapes
        # print (f"X_all_coords: {X_all_coords.shape}, node_type: {node_type.shape}, node_s: {node_s.shape}, node_v: {node_v.shape}, edge_index: {edge_index.shape}, edge_s: {edge_s.shape}, edge_v: {edge_v.shape}")
        # printing first few rows of each tensor for debugging
        # print("First few rows of tensors:")
        # print("X_all_coords:", X_all_coords[:5])
        # print("node_type:", node_type[:5])
        # print("node_s:", node_s[:5])
        # print("node_v:", node_v[:5])
        # print("edge_index:", edge_index[:,:5])
        # print("edge_s:", edge_s[:5])
        # print("edge_v:", edge_v[:5])
        edge_index=edge_index.long()
        return Data(
            x=X_all_coords, node_type=node_type,
            node_s=torch.nan_to_num(node_s), node_v=torch.nan_to_num(node_v),
            edge_s=torch.nan_to_num(edge_s), edge_v=torch.nan_to_num(edge_v),
            edge_index=edge_index
        )

    def featurize_graph_with_water(self, protein: Protein, center=None, crop_size=None,
                                   rbf_D_count:int = 16, positional_emb_dim: int =16) -> Data:
        X_res_all_coords, res_list = stack_residue_coordinates(protein) # All atom coordinates for all residues
        if X_res_all_coords.numel() == 0 and not protein.read_water: # Handle empty protein
             return None 
            #  return Data(x=torch.empty(0,3), node_type=torch.empty(0, dtype=torch.long), node_s=torch.empty(0,6), node_v=torch.empty(0,2,3,3), edge_index=torch.empty(2,0), edge_s=torch.empty(0,32), edge_v=torch.empty(0,3))
 
        X_res_ca = X_res_all_coords[1::3] if X_res_all_coords.size(0) >=2 else X_res_all_coords # CA atoms
        
        X_water_coords = stack_water_coordinates(protein)
        has_water = X_water_coords is not None and X_water_coords.numel() > 0

        # Initial indices before cropping
        original_res_indices = torch.arange(X_res_ca.size(0))
        original_water_indices = torch.arange(X_water_coords.size(0)) if has_water else torch.empty((0,), dtype=torch.long)

        # Cropping
        if center is not None and crop_size is not None:
            if X_res_ca.numel() > 0:
                mask_res_ca = ((X_res_ca - center).abs() < crop_size / 2).all(dim=-1)
                X_res_ca = X_res_ca[mask_res_ca]
                keep_res_idx = original_res_indices[mask_res_ca] # Map cropped indices back to original X_res_all_coords indices for feature slicing
            else:
                keep_res_idx = torch.empty((0,), dtype=torch.long)

            if has_water and X_water_coords.numel() > 0:
                mask_water = ((X_water_coords - center).abs() < crop_size / 2).all(dim=-1)
                X_water_coords = X_water_coords[mask_water]
                # keep_water_idx = original_water_indices[mask_water] # If water features were derived from an 'all_water' set
            else:
                has_water = False # Update if cropping removed all water or no water to begin with
        else:
            keep_res_idx = original_res_indices

        X_all_coords = X_res_ca
        if has_water and X_water_coords.numel() > 0:
            X_all_coords = torch.cat([X_res_ca, X_water_coords], dim=0)
        
        if X_all_coords.numel() == 0: # Handle case where cropping results in no atoms
            return None
            # return Data(x=torch.empty(0,3), node_type=torch.empty(0, dtype=torch.long), node_s=torch.empty(0,6), node_v=torch.empty(0,2,3,3), edge_index=torch.empty(2,0), edge_s=torch.empty(0,32), edge_v=torch.empty(0,3))

        # Node types
        node_type_res = torch.zeros(X_res_ca.size(0), dtype=torch.long)
        if has_water and X_water_coords.numel() > 0:
            node_type_water = torch.ones(X_water_coords.size(0), dtype=torch.long)
            node_type = torch.cat([node_type_res, node_type_water])
        else:
            node_type = node_type_res

        # Node scalar features
        node_s_all_residues = get_residue_dihedrals(X_res_all_coords)
        node_s_res = node_s_all_residues[keep_res_idx] if node_s_all_residues.numel() > 0 else torch.empty((0, node_s_all_residues.size(1) if node_s_all_residues.ndim > 1 else 6 ))
        node_aa_features = get_aa_one_hot(protein) # One-hot encoding of amino acids
        node_aa_features = node_aa_features[keep_res_idx] if node_aa_features is not None and node_aa_features.numel() > 0 else torch.empty((0, node_aa_features.size(1) if node_aa_features is not None else 20)) # Assuming 20 amino acids
        # print (f"Node AA features shape: {node_aa_features.shape}") # Debugging line
        # print (f"Node scalar features shape before concatenation: {node_s_res.shape}") # Debugging line
        if node_aa_features is not None and node_aa_features.numel() > 0:
            node_s_res = torch.cat([node_s_res, node_aa_features], dim=1) if node_s_res.numel() > 0 else node_aa_features
        node_s = node_s_res
        if has_water and X_water_coords.numel() > 0:
            num_res_scalar_feats = node_s_res.size(1) if node_s_res.numel() > 0 else 6 # Default to 6 if no residues
            node_s_water = get_water_embeddings(X_water_coords, num_embeddings=num_res_scalar_feats)
            node_s = torch.cat([node_s_res, node_s_water], dim=0)

        # Node vector features
        sidechain_orient_all = get_sidechain_orientation(X_res_all_coords)
        backbone_orient_all = get_backbone_orientation(X_res_all_coords)
        
        if sidechain_orient_all.numel() > 0 and backbone_orient_all.numel() > 0:
             node_v_all_residues = torch.cat([sidechain_orient_all.unsqueeze(-2), backbone_orient_all], dim=-2) # Dim becomes [N_res, 1+2, 3, 3]
             node_v_res = node_v_all_residues[keep_res_idx]
        elif sidechain_orient_all.numel() > 0: # Only sidechain
             node_v_all_residues = sidechain_orient_all.unsqueeze(-2) # [N_res, 1, 3, 3]
             node_v_res = node_v_all_residues[keep_res_idx]
        elif backbone_orient_all.numel() > 0: # Only backbone
             node_v_all_residues = backbone_orient_all # [N_res, 2, 3, 3] assuming this shape
             node_v_res = node_v_all_residues[keep_res_idx]
        else: 
            node_v_res = torch.empty((0, 1, 3, 3), dtype=torch.float32)

        node_v = node_v_res
        if has_water and X_water_coords.numel() > 0:
            # now it's just zeros, but could be replaced with water orientation if available
            num_feats = node_v_res.size(1) 
            feats_dim = node_v_res.size(2) 
            node_v_water = torch.zeros(X_water_coords.size(0), num_feats, feats_dim) # Water nodes typically have no orientation
            node_v = torch.cat([node_v_res, node_v_water], dim=0)

        # == Edges ==
        # edge_index = torch_cluster.knn_graph(X_all_coords, k=min(self.top_k, X_all_coords.size(0)-1) if X_all_coords.size(0)>1 else 0)
        if X_all_coords.size(0) > 1 and self.top_k > 0:
            k = min(self.top_k, X_all_coords.size(0) - 1)
            edge_index = torch_cluster.knn_graph(X_all_coords, k=k)
        else:
            return None
        E_vectors = X_all_coords[edge_index[0]] - X_all_coords[edge_index[1]] if edge_index.numel() > 0 else torch.empty((0,3))
        edge_v_2D = _normalize_torch(E_vectors) if E_vectors.numel() > 0 else torch.empty((0,3))
        edge_v = edge_v_2D.unsqueeze(1) if edge_v_2D.numel() > 0 else torch.empty((0,1,3))

        edge_dist = E_vectors.norm(dim=-1) if E_vectors.numel() > 0 else torch.empty((0))

        rbf_features = get_rbf(edge_dist, D_count=rbf_D_count, device=edge_index.device) if edge_dist.numel() > 0 else torch.empty((0,16))
        pos_emb_features = get_positional_embeddings(edge_index, num_embeddings=positional_emb_dim) if edge_index.numel() > 0 else torch.empty((0,16))
        edge_s = torch.cat([rbf_features, pos_emb_features], dim=-1)
        edge_index=edge_index.long()
        
        return Data(
            x=X_all_coords, node_type=node_type,
            node_s=torch.nan_to_num(node_s), node_v=torch.nan_to_num(node_v),
            edge_s=torch.nan_to_num(edge_s), edge_v=torch.nan_to_num(edge_v),
            edge_index=edge_index
        )

    def featurize_graph_with_virtual_nodes(
        self,
        protein_w_water: Protein, # This was 'protein_w' or 'protein' arg in dataset method
        protein_wo_water: Protein,  # This was 'self.protein' from dataset
        ligand: Ligand,    # This was 'self.ligand' from dataset
        ligand_for_vn: Ligand = None, # Optional ligand for virtual node generation
        center=None,
        crop_size=None,
        target_name: str = None, # Optional target name for logging
        rbf_D_count: int = 16, # Default RBF count
        positional_emb_dim: int = 16 # Default positional embedding dimension
    ) -> Data:
        X_res_all, res_list = stack_residue_coordinates(protein_w_water) # this already excludes water
        # get sidechain atom coordinates - THIS PART NEEDS TO BE CHECKED. 
        X_sidechain_all = []
        sidechain_lens = []  
        for residue in res_list:
            coords = []
            for atm in residue.atoms:
                if atm.name not in ['N', 'CA', 'C']:  # Exclude backbone atoms
                    coords.append(atm.coordinates)
            if coords:
                coords = np.stack(coords)
            else:
                coords = np.empty((0, 3))
            X_sidechain_all.append(torch.from_numpy(coords).float())
            sidechain_lens.append(coords.shape[0])

        # Padding - fit longest sidechain (shape: [N_residues, max_atoms, 3])
        max_atoms = max(sidechain_lens)
        N_residues = len(res_list)
        X_sidechain_padded = torch.zeros((N_residues, max_atoms, 3), dtype=torch.float32)

        for i, coords in enumerate(X_sidechain_all):
            if coords.shape[0] > 0:
                X_sidechain_padded[i, :coords.shape[0], :] = coords

        # mask for padding (0: padded) 
        X_sidechain_mask = torch.zeros((N_residues, max_atoms), dtype=torch.bool)
        for i, length in enumerate(sidechain_lens):
            if length > 0:
                X_sidechain_mask[i, :length] = 1

        # ca atoms
        X_res_ca = X_res_all[1::3]  # assuming every 3rd atom is CA 
        # X_res_ca shape: torch.Size([807, 3]) / X_sidechain_padded shape: torch.Size([807, 14, 3])
        X_water= stack_water_coordinates(protein_w_water)
        has_water = X_water is not None and X_water.numel() > 0
        if ligand_for_vn is not None: 
            virtual_nodes_list = generate_virtual_nodes(protein_wo_water, ligand_for_vn, only_backbone=True) # clash N,CA,C,CB
        else:
            virtual_nodes_list = generate_virtual_nodes(protein_wo_water, ligand, only_backbone=True) # clash N,CA,C,CB
        if virtual_nodes_list: # Check if list is not empty
            if target_name:
                filepath = f'/home.galaxy4/j2ho/projects/aff-yip/vn_temp/{target_name}.pdb'
                # write_virtual_nodes_pdb(virtual_nodes_list, filepath=filepath, element='C', chain_id='X')
            X_virtual = torch.from_numpy(np.stack([v.coordinates for v in virtual_nodes_list])).float()
        else:
            X_virtual = torch.empty((0,3), dtype=torch.float32)
        # Water occupancy for virtual nodes (if applicable from your VirtualNode structure)
        if has_water:
            for v_node in virtual_nodes_list:
                v_node.set_water_occupancy(X_water) 

        # Initial indices
        original_res_indices = torch.arange(X_res_ca.size(0))
        original_virtual_indices = torch.arange(X_virtual.size(0))
        # Cropping
        if center is not None and crop_size is not None:
            mask_res_ca = ((X_res_ca - center).abs() < crop_size / 2).all(dim=-1)
            X_res_ca = X_res_ca[mask_res_ca]
            keep_res_idx = original_res_indices[mask_res_ca]
            # keep sidechain coordinates based on residue CA mask
            X_sidechain_padded = X_sidechain_padded[mask_res_ca]
            X_sidechain_mask = X_sidechain_mask[mask_res_ca]

            if has_water:
                mask_water = ((X_water- center).abs() < crop_size / 2).all(dim=-1)
                X_water= X_water[mask_water]

            if X_virtual.numel() > 0:
                mask_virtual = ((X_virtual - center).abs() < crop_size / 2).all(dim=-1)
                X_virtual = X_virtual[mask_virtual]
                keep_virtual_idx = original_virtual_indices[mask_virtual]
            else:
                keep_virtual_idx = torch.empty((0,),dtype=torch.long)
        else:
            keep_res_idx = original_res_indices
            keep_virtual_idx = original_virtual_indices # All virtual nodes kept
        # now X_res_ca, X_water, X_virtual are cropped based on the mask
        # Concatenate all coordinates for graph
        X_all_coords_list = [X_res_ca]
        if has_water: X_all_coords_list.append(X_water)
        if X_virtual.numel() > 0: X_all_coords_list.append(X_virtual)
        if not X_all_coords_list or all(c.numel()==0 for c in X_all_coords_list): # All components are empty
            return Data(x=torch.empty(0,3), node_type=torch.empty(0, dtype=torch.long), node_s=torch.empty(0,6), node_v=torch.empty(0,2,3,3), edge_index=torch.empty(2,0), edge_s=torch.empty(0,32), edge_v=torch.empty(0,3), feature_mask=torch.empty(0,6))
        X_all = torch.cat(X_all_coords_list, dim=0) # Concatenate all coordinates
        # Node types
        node_type_list = [torch.zeros(X_res_ca.size(0), dtype=torch.long)]
        if has_water: node_type_list.append(torch.ones(X_water.size(0), dtype=torch.long))
        if X_virtual.numel() > 0: node_type_list.append(2 * torch.ones(X_virtual.size(0), dtype=torch.long))
        node_type = torch.cat([nt for nt in node_type_list if nt.numel() > 0])
        # Node scalar features
        node_s_all_residues = get_residue_dihedrals(X_res_all) # From original, uncropped full atom protein
        node_s_res = node_s_all_residues[keep_res_idx] if node_s_all_residues.numel() > 0 else torch.empty((0, node_s_all_residues.size(1) if node_s_all_residues.ndim > 1 else 6 ))
        node_aa_features = get_aa_one_hot(protein_wo_water) # One-hot encoding of amino acids
        node_aa_features = node_aa_features[keep_res_idx] if node_aa_features is not None and node_aa_features.numel() > 0 else torch.empty((0, node_aa_features.size(1) if node_aa_features is not None else 20)) # Assuming 20 amino acids
        if node_aa_features is not None and node_aa_features.numel() > 0:
            node_s_res = torch.cat([node_s_res, node_aa_features], dim=1) if node_s_res.numel() > 0 else node_aa_features
        node_s_list = [node_s_res]
        num_scalar_features = node_s_res.size(1) # if node_s_res.numel() > 0 else 6 # Default from dihedrals
        if has_water > 0:
            node_s_water = get_water_embeddings(X_water, num_embeddings=num_scalar_features)
            node_s_list.append(node_s_water)
        if X_virtual.numel() > 0:
            # Virtual node scalar features: first is occupancy, rest are zeros
            # This assumes virtual_nodes_list is now cropped or v.water_occupancy is indexed by keep_virtual_idx
            cropped_virtual_nodes = [virtual_nodes_list[i] for i in keep_virtual_idx.tolist()] if virtual_nodes_list else []

            if cropped_virtual_nodes and hasattr(cropped_virtual_nodes[0], 'water_occupancy'):
                 virtual_occupancies = torch.tensor([v.water_occupancy for v in cropped_virtual_nodes], dtype=torch.float32).unsqueeze(-1)
            else: # If no occupancy data, default to zeros ??? 
                 virtual_occupancies = torch.zeros(X_virtual.size(0), 1, dtype=torch.float32)
            node_s_virtual_padding = torch.zeros(X_virtual.size(0), num_scalar_features - 1)
            node_s_virtual = torch.cat([virtual_occupancies, node_s_virtual_padding], dim=1)
            node_s_list.append(node_s_virtual)
        node_s = torch.cat([ns for ns in node_s_list if ns.numel() > 0], dim=0) if any(ns.numel() > 0 for ns in node_s_list) else torch.empty((0, num_scalar_features))

        # Node vector features (similar logic to scalar features for concatenation)
        sidechain = get_sidechain_orientation(X_res_all) # (num_residues, 3)
        sidechain = sidechain.unsqueeze(-2) # (num_residues, 1, 3)
        backbone = get_backbone_orientation(X_res_all) # (num_residues, 2, 3)
        if sidechain.numel() > 0 and backbone.numel() > 0 :
             node_v_all_residues = torch.cat([sidechain, backbone], dim=-2)
             node_v_res = node_v_all_residues[keep_res_idx]
        elif sidechain.numel() > 0:
             node_v_all_residues = sidechain
             node_v_res = node_v_all_residues[keep_res_idx]
        elif backbone.numel() > 0:
             node_v_all_residues = backbone
             node_v_res = node_v_all_residues[keep_res_idx]
        else: 
            node_v_res = torch.empty((0, 1, 3, 3), dtype=torch.float32) # tensor([], size=(0, 1, 3, 3))
        
        num_feats = node_v_res.size(1) # if node_v_res.numel() > 0 else default_orient_dim
        feats_dim = node_v_res.size(2) # if node_v_res.numel() > 0 else default_vec_dim

        node_v_list = [node_v_res]
        if has_water:
            node_v_list.append(torch.zeros(X_water.size(0), num_feats, feats_dim)) # Water nodes typically have no orientation
        if X_virtual.numel() > 0:
            node_v_list.append(torch.zeros(X_virtual.size(0), num_feats, feats_dim)) # Virtual nodes have no orientation
        node_v = torch.cat([nv for nv in node_v_list if nv.numel() > 0], dim=0) if any(nv.numel() > 0 for nv in node_v_list) else torch.empty((0, num_feats, feats_dim))
        # Edges / reminder: X_all is now concatenated from residues, water, and virtual nodes, cropped
        edge_index = torch_cluster.knn_graph(X_all, k=min(self.top_k, X_all.size(0)-1) if X_all.size(0)>1 else 0)
        E_vectors = X_all[edge_index[0]] - X_all[edge_index[1]] if edge_index.numel() > 0 else torch.empty((0,3))
        # edge_v = _normalize_torch(E_vectors) if E_vectors.numel() > 0 else torch.empty((0,3))
        edge_v_2D = _normalize_torch(E_vectors) if E_vectors.numel() > 0 else torch.empty((0,3))
        edge_v = edge_v_2D.unsqueeze(1) if edge_v_2D.numel() > 0 else torch.empty((0,1,3))

        edge_dist = E_vectors.norm(dim=-1) if E_vectors.numel() > 0 else torch.empty((0))
        rbf_features = get_rbf(edge_dist, D_count=rbf_D_count, device=edge_index.device) if edge_dist.numel() > 0 else torch.empty((0,16))
        pos_emb_features = get_positional_embeddings(edge_index, num_embeddings=positional_emb_dim) if edge_index.numel() > 0 else torch.empty((0,16))
        edge_s_geom = torch.cat([rbf_features, pos_emb_features], dim=-1)

        # Edge types (protein-protein, protein-water, water-virtual, etc.)
        num_node_categories = 3 # 0:protein, 1:water, 2:virtual
        edge_s = edge_s_geom
        if edge_index.numel() > 0 and node_type.numel() > 0:
            src_type = node_type[edge_index[0]]
            dst_type = node_type[edge_index[1]]
            edge_type_id = src_type * num_node_categories + dst_type # Max id will be num_node_categories*num_node_categories - 1
            edge_type_onehot = F.one_hot(edge_type_id, num_classes=num_node_categories**2).float()
            edge_s = torch.cat([edge_s_geom, edge_type_onehot], dim=-1)
        else: # if no edges, edge_s might need to match expected feature dim if it includes edge_type_onehot
            edge_s = torch.cat([edge_s_geom, torch.empty(edge_s_geom.size(0), num_node_categories**2)], dim=-1)


        # Feature mask for Virtual Nodes
        feature_mask = torch.ones_like(node_s)
        if X_virtual.numel() > 0 and node_s.numel() > 0: # Only apply if virtual nodes and scalar features exist
            num_virtual_nodes_in_graph = X_virtual.size(0)
            # Ensure indexing doesn't go out of bounds if node_s became smaller than expected
            if feature_mask.size(0) >= num_virtual_nodes_in_graph:
                 feature_mask[-num_virtual_nodes_in_graph:, 1:] = 0 # Virtual nodes only have first scalar feature valid (occupancy) 
        # print (f'protein node count: {X_res_ca.size(0)}, water node count: {X_water.size(0)}, virtual node count: {X_virtual.size(0)}')
        # print (f'sidechain padded shape: {X_sidechain_padded.shape}, sidechain mask shape: {X_sidechain_mask.shape}')
        return Data(
            x=X_all, node_type=node_type,
            node_s=torch.nan_to_num(node_s), feature_mask=feature_mask,
            node_v=torch.nan_to_num(node_v),
            edge_s=torch.nan_to_num(edge_s), edge_v=torch.nan_to_num(edge_v),
            edge_index=edge_index, 
            res_list=res_list, # Include residue list for reference
            X_sidechain_padded=X_sidechain_padded, # Include padded sidechain coordinates
            X_sidechain_mask=X_sidechain_mask # Include sidechain mask for padding
        )
 