
import os
import torch
import torch_geometric
import torch_cluster
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset

from src.data.featurizers.utils import stack_residue_coordinates, get_residue_dihedrals, get_sidechain_orientation, get_backbone_orientation, \
                        _normalize_torch, get_rbf, get_positional_embeddings, stack_water_coordinates, \
                        generate_virtual_nodes, get_water_embeddings

from src.data.structures import Protein, Ligand, VirtualNode
from src.io.write_pdb import write_virtual_nodes_pdb

class ProteinDataset(Dataset):
    def __init__(self, data_path, target_dict=None, mode='train', top_k=30, max_length=500):
        self.data_path = data_path
        self.mode = mode
        self.top_k = top_k
        self.samples = []

        self.protein_w = Protein(
            pdb_filepath=os.path.join(data_path, '5hz8.pdb'),
            read_water=True, read_ligand=False
        )
        self.protein = Protein(
            pdb_filepath=os.path.join(data_path, '5hz8.pdb'),
            read_water=False, read_ligand=False
        )
        self.ligand = Ligand(
            mol2_filepath='/home/j2ho/DB/pdbbind/v2020-refined/5hz8/5hz8_ligand.mol2'
        )
        if target_dict: 
            for target in target_dict:
                try:
                    pdbfile = target.get('pdb_file', None)
                    ligfile = target.get('ligand_file', None)
                    affinity = target.get('affinity', None)
                    protein_w = Protein(pdb_filepath=pdbfile, read_water=True, read_ligand=False)
                    protein = Protein(pdb_filepath=pdbfile, read_water=False, read_ligand=False)
                    ligand = Ligand(mol2_filepath=ligfile, drop_H=False)  
                    # with virtual nodes and water
                    protein_virtual_graph = self.featurize_protein_graph_with_virtual_nodes(
                        protein_w) #center=center, crop_size=30.0) 
                    # without virtual nodes and water
                    protein_graph = self.featurize_protein_graph(protein)
                    ligand_graph = self.featurize_ligand_graph(ligand)  # your implementation
                    self.samples.append({
                    'name': target.get('name', 'None'),
                    'protein_virtual': protein_virtual_graph,
                    'protein': protein_graph,
                    'ligand': ligand_graph,
                    'affinity': affinity,
                    'ligand_coords': ligand.get_coordinates()
                })

                except Exception as e:
                    print(f"[Warning] Skipped {target}: {e}")
        else:
            # Testing mode with default protein and ligand
            center = torch.tensor([-4.374, -7.353, -19.189], dtype=torch.float32)
            protein_virtual_graph = self.featurize_protein_graph_with_virtual_nodes(
                self.protein_w, center=center, crop_size=30.0
            )
            protein_graph = self.featurize_protein_graph(self.protein, center=center, crop_size=30.0)
            ligand_graph = self.featurize_ligand_graph(self.ligand)  # your implementation
            data = {
                'name': 'default_protein',
                'protein': protein_graph,
                'ligand': ligand_graph,
                'affinity': None,  # No affinity available
                'ligand_coords': self.ligand.get_coordinates()
            }
            self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def featurize_protein_graph_nowater(self, protein: Protein, center=None, crop_size=None) -> torch_geometric.data.Data:
        # === Full coordinates ===
        X_res_all = stack_residue_coordinates(protein)
        X_res = X_res_all[1::3]  # CA only
        X_all = X_res
        # === Optional cropping ===
        if center is not None and crop_size is not None:
            mask_res = ((X_res - center).abs() < crop_size / 2).all(dim=-1)

            X_res = X_res[mask_res]
            X_all = X_res
            keep_res_idx = mask_res.nonzero(as_tuple=False).squeeze(-1)
        else: # No cropping
            keep_res_idx = torch.arange(X_res.size(0))
        # === Node type ===
        node_type = torch.cat([
            torch.zeros(X_res.size(0)), torch.ones(X_water.size(0))
        ]).long()

        # === Node scalar features ===
        node_s_all_res = get_residue_dihedrals(X_res_all)
        node_s = node_s_all_res[keep_res_idx]

        # === Node vector features ===
        sidechain = get_sidechain_orientation(X_res_all).unsqueeze(-2)
        backbone = get_backbone_orientation(X_res_all)
        node_v_res_all = torch.cat([sidechain, backbone], dim=-2)
        node_v = node_v_res_all[keep_res_idx]

        # === Edge index and edge features ===
        edge_index = torch_cluster.knn_graph(X_all, k=self.top_k)
        E_vectors = X_all[edge_index[0]] - X_all[edge_index[1]]
        edge_v = _normalize_torch(E_vectors)
        edge_dist = E_vectors.norm(dim=-1)
        edge_s = torch.cat([
            get_rbf(edge_dist, D_count=16, device=edge_index.device),
            get_positional_embeddings(edge_index, num_embeddings=16)
        ], dim=-1)

        data = torch_geometric.data.Data(
            x=X_all, node_type=node_type,
            node_s=torch.nan_to_num(node_s),
            node_v=torch.nan_to_num(node_v),
            edge_s=torch.nan_to_num(edge_s),
            edge_v=torch.nan_to_num(edge_v),
            edge_index=edge_index
        )

        return data
    
    def featurize_protein_graph(self, protein: Protein, center=None, crop_size=None) -> torch_geometric.data.Data:
        # === Full coordinates ===
        X_res_all = stack_residue_coordinates(protein)
        X_res = X_res_all[1::3]  # CA atoms only
        X_water = stack_water_coordinates(protein)
        has_water = X_water is not None and X_water.numel() > 0

        X_all = X_res if not has_water else torch.cat([X_res, X_water], dim=0)

        print(f"Protein featurization: {X_res.size(0)} residues, {X_water.size(0) if has_water else 0} water molecules")

        # === Optional cropping ===
        if center is not None and crop_size is not None:
            mask_res = ((X_res - center).abs() < crop_size / 2).all(dim=-1)
            X_res = X_res[mask_res]
            keep_res_idx = mask_res.nonzero(as_tuple=False).squeeze(-1)

            if has_water:
                mask_water = ((X_water - center).abs() < crop_size / 2).all(dim=-1)
                X_water = X_water[mask_water]
                keep_water_idx = mask_water.nonzero(as_tuple=False).squeeze(-1)
                X_all = torch.cat([X_res, X_water], dim=0)
            else:
                keep_water_idx = torch.empty((0,), dtype=torch.long)
                X_all = X_res
        else:
            keep_res_idx = torch.arange(X_res.size(0))
            keep_water_idx = torch.arange(X_water.size(0)) if has_water else torch.empty((0,), dtype=torch.long)

        print(f"After cropping: {X_res.size(0)} residues, {X_water.size(0) if has_water else 0} water molecules")

        # === Node types ===
        if has_water:
            node_type = torch.cat([
                torch.zeros(X_res.size(0)),
                torch.ones(X_water.size(0))
            ]).long()
        else:
            node_type = torch.zeros(X_res.size(0)).long()

        # === Node scalar features ===
        node_s_all_res = get_residue_dihedrals(X_res_all)
        node_s_res = node_s_all_res[keep_res_idx]

        if has_water:
            # node_s_water = torch.rand(X_water.size(0), node_s_res.size(1))  # placeholder
            node_s_water = get_water_embeddings(X_water, num_embeddings=node_s_res.size(1))  # X_water is already cropped
            node_s = torch.cat([node_s_res, node_s_water], dim=0)
        else:
            node_s = node_s_res

        # === Node vector features ===
        sidechain = get_sidechain_orientation(X_res_all).unsqueeze(-2)
        backbone = get_backbone_orientation(X_res_all)
        node_v_res_all = torch.cat([sidechain, backbone], dim=-2)
        node_v_res = node_v_res_all[keep_res_idx]

        if has_water:
            node_v_water = torch.zeros(X_water.size(0), 3, 3)
            node_v = torch.cat([node_v_res, node_v_water], dim=0)
        else:
            node_v = node_v_res

        # === Edge index and edge features ===
        edge_index = torch_cluster.knn_graph(X_all, k=self.top_k)
        E_vectors = X_all[edge_index[0]] - X_all[edge_index[1]]
        edge_v = _normalize_torch(E_vectors)
        edge_dist = E_vectors.norm(dim=-1)
        edge_s = torch.cat([
            get_rbf(edge_dist, D_count=16, device=edge_index.device),
            get_positional_embeddings(edge_index, num_embeddings=16)
        ], dim=-1)

        # === Final Data object ===
        data = torch_geometric.data.Data(
            x=X_all, node_type=node_type,
            node_s=torch.nan_to_num(node_s),
            node_v=torch.nan_to_num(node_v),
            edge_s=torch.nan_to_num(edge_s),
            edge_v=torch.nan_to_num(edge_v),
            edge_index=edge_index
        )

        return data

    def featurize_protein_graph_with_virtual_nodes(self, protein: Protein, center=None, crop_size=None) -> torch_geometric.data.Data:
        # Featurize the protein graph with virtual nodes
        # === Full coordinates ===
        X_res_all = stack_residue_coordinates(protein)
        X_res = X_res_all[1::3]  # CA only
        X_water = stack_water_coordinates(protein)
        has_water = X_water is not None and X_water.numel() > 0

        virtual_nodes = generate_virtual_nodes(self.protein, self.ligand)
        # write 
        write_virtual_nodes_pdb(virtual_nodes, filepath='virtual_nodes.pdb', element='C', chain_id='X')
        # coordinates_list = [v.coordinates for v in virtual_nodes]
        # X_virtual = torch.from_numpy(np.array(coordinates_list)).float()
        X_virtual = torch.from_numpy(np.stack([v.coordinates for v in virtual_nodes])).float()

        if has_water:
            for v in virtual_nodes:
                v.set_water_occupancy(X_water)
            X_all = torch.cat([X_res, X_water, X_virtual], dim=0)
            """ 
            나중에는 X_water 다 쓰지 않고 virtual node만 쓰고, X_water 중에서 중요한 explicit water있으면 그것만 따로 받아서
            """
        else:
            X_all = torch.cat([X_res, X_virtual], dim=0)

        print(f"Protein featurization: {X_res.size(0)} residues, {X_water.size(0) if has_water else 0} water molecules, {X_virtual.size(0)} virtual nodes")

        # === Optional cropping ===
        if center is not None and crop_size is not None:
            mask_res = ((X_res - center).abs() < crop_size / 2).all(dim=-1)
            X_res = X_res[mask_res]
            keep_res_idx = mask_res.nonzero(as_tuple=False).squeeze(-1)

            if has_water:
                mask_water = ((X_water - center).abs() < crop_size / 2).all(dim=-1)
                X_water = X_water[mask_water]
                keep_water_idx = mask_water.nonzero(as_tuple=False).squeeze(-1)
            else:
                keep_water_idx = torch.empty((0,), dtype=torch.long)

            mask_virtual = ((X_virtual - center).abs() < crop_size / 2).all(dim=-1)
            X_virtual = X_virtual[mask_virtual]
            keep_virtual_idx = mask_virtual.nonzero(as_tuple=False).squeeze(-1)
            if has_water:
                X_all = torch.cat([X_res, X_water, X_virtual], dim=0)
            else:
                X_all = torch.cat([X_res, X_virtual], dim=0)
        else:
            keep_res_idx = torch.arange(X_res.size(0))
            keep_water_idx = torch.arange(X_water.size(0)) if has_water else torch.empty((0,), dtype=torch.long)

        print(f"After cropping: {X_res.size(0)} residues, {X_water.size(0) if has_water else 0} water molecules, {X_virtual.size(0)} virtual nodes")

        # === Node type ===
        if has_water:
            node_type = torch.cat([
                torch.zeros(X_res.size(0)),
                torch.ones(X_water.size(0)),
                2 * torch.ones(X_virtual.size(0))
            ]).long()
        else:
            node_type = torch.cat([
                torch.zeros(X_res.size(0)),
                2 * torch.ones(X_virtual.size(0))
            ]).long()

        # === Node scalar features ===
        node_s_all_res = get_residue_dihedrals(X_res_all)
        node_s_res = node_s_all_res[keep_res_idx]

        if has_water:
            node_s_water = get_water_embeddings(X_water, num_embeddings=node_s_res.size(1))
            virtual_occupancies = torch.from_numpy(np.stack([v.water_occupancy for v in virtual_nodes])).float()
            # keep virtual 
            virtual_occupancies = virtual_occupancies[keep_virtual_idx]
            node_s_virtual = torch.cat([virtual_occupancies.unsqueeze(-1), 
                                        torch.zeros(X_virtual.size(0), node_s_res.size(1) - 1)], dim=1)
            node_s = torch.cat([node_s_res, node_s_water, node_s_virtual], dim=0) # all with feature dimension 6 

        else:
            node_s = torch.cat([node_s_res, torch.zeros(X_virtual.size(0), node_s_res.size(1))], dim=0)

        print(f"Node scalar features: {node_s.size(0)} nodes")

        # === Node vector features ===
        sidechain = get_sidechain_orientation(X_res_all).unsqueeze(-2)
        backbone = get_backbone_orientation(X_res_all)
        node_v_res_all = torch.cat([sidechain, backbone], dim=-2)
        node_v_res = node_v_res_all[keep_res_idx]

        if has_water:
            node_v_water = torch.zeros(X_water.size(0), 3, 3)
            node_v = torch.cat([node_v_res, node_v_water, torch.zeros(X_virtual.size(0), 3, 3)], dim=0)
        else:
            node_v = torch.cat([node_v_res, torch.zeros(X_virtual.size(0), 3, 3)], dim=0)

        # === Edge index and edge features ===
        edge_index = torch_cluster.knn_graph(X_all, k=self.top_k)
        E_vectors = X_all[edge_index[0]] - X_all[edge_index[1]]
        edge_v = _normalize_torch(E_vectors)
        edge_dist = E_vectors.norm(dim=-1)
        edge_s = torch.cat([
            get_rbf(edge_dist, D_count=16, device=edge_index.device),
            get_positional_embeddings(edge_index, num_embeddings=16)
        ], dim=-1)

        # === Edge types ===
        src_type = node_type[edge_index[0]]
        dst_type = node_type[edge_index[1]]
        edge_type_id = src_type * 3 + dst_type
        edge_type_onehot = torch.nn.functional.one_hot(edge_type_id, num_classes=9).float()
        edge_s = torch.cat([edge_s, edge_type_onehot], dim=-1)

        # === Feature mask for Virtual Nodes ===
        feature_mask = torch.ones_like(node_s)
        feature_mask[-X_virtual.size(0):, 1:] = 0  # Virtual nodes only have first feature valid

        data = torch_geometric.data.Data(
            x=X_all, node_type=node_type,
            node_s=torch.nan_to_num(node_s),
            feature_mask=feature_mask,
            node_v=torch.nan_to_num(node_v),
            edge_s=torch.nan_to_num(edge_s),
            edge_v=torch.nan_to_num(edge_v),
            edge_index=edge_index
        )

        return data

    def featurize_ligand_graph(self, ligand: Ligand) -> torch_geometric.data.Data:
        # Placeholder for ligand graph featurization
        # This should be implemented based on the specific requirements for ligand representation
        X = torch.tensor(ligand.get_coordinates(), dtype=torch.float32)
        node_type = torch.zeros(X.size(0), dtype=torch.long)
        bonds = ligand.bonds
        # bonds = 
        atoms = ligand.atoms

class ProteinCollator:
    """
    Collates protein samples into a batch
    """
    def __init__(self, has_quality=False):
        self.has_quality = has_quality
    
    def __call__(self, batch):
        """
        Collate protein samples
        :param batch: List of protein samples
        :return: Batched tensors
        """
        if self.has_quality:
            X, S, mask, y = zip(*batch)
        else:
            X, S, mask = zip(*batch)
        
        # Get sequence lengths
        lengths = [x.shape[0] for x in X]
        max_length = max(lengths)
        
        # Pad the sequences
        X_padded = []
        S_padded = []
        mask_padded = []
        
        for i, l in enumerate(lengths):
            # Padding for X
            padding = torch.zeros(max_length - l, 4, 3)
            X_padded.append(torch.cat([X[i], padding], dim=0))
            
            # Padding for S
            padding = torch.zeros(max_length - l, dtype=torch.long)
            S_padded.append(torch.cat([S[i], padding], dim=0))
            
            # Padding for mask
            padding = torch.zeros(max_length - l, dtype=torch.bool)
            mask_padded.append(torch.cat([mask[i], padding], dim=0))
        
        # Stack
        X_batch = torch.stack(X_padded, dim=0)
        S_batch = torch.stack(S_padded, dim=0)
        mask_batch = torch.stack(mask_padded, dim=0)
        
        if self.has_quality:
            y_batch = torch.tensor(y, dtype=torch.float32)
            return X_batch, S_batch, mask_batch, y_batch
        else:
            return X_batch, S_batch, mask_batch

    
if __name__ == '__main__':
    # Example usage
    ProteinDataset = ProteinDataset(data_path='.', mode='train')