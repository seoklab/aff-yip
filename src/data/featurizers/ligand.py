# src/data/featurizers/ligand.py
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np

from ..structures import Ligand, Atom # For type hinting and potentially creating dummy atom

class LigandFeaturizer:
    def __init__(self):
        # Define vocabularies and mappings
        # Ensure 'H' is included if not dropped, and handle other common elements
        self.element_vocab = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H', 'B', 'SI', 'SE', 'NA', 'K', 'MG', 'CA', 'ZN', 'MN', 'UNK_ELEM']
        self.element_to_index = {elem: i for i, elem in enumerate(self.element_vocab)}

        # Example SYBYL atom types. Populate this based on your data.
        self.atom_type_vocab = ['C.3', 'C.2', 'C.ar', 'C.cat', 'N.3', 'N.2', 'N.ar', 'N.am', 'N.pl3', 'N.4',
                                'O.3', 'O.2', 'O.co2', 'S.3', 'S.O2', 'P.3', 'H', 'LP', 'DU', 'ANY', 'UNK_ATOM_TYPE']
        self.atom_type_to_index = {atype: i for i, atype in enumerate(self.atom_type_vocab)}

        self.bond_type_vocab = ['1', '2', '3', 'ar', 'am', 'du', 'un', 'nc', 'UNK_BOND'] # MOL2 bond types
        self.bond_type_to_index = {btype: i for i, btype in enumerate(self.bond_type_vocab)}
        self.num_bond_features = len(self.bond_type_vocab)

        # Determine feature dimension for empty atom list case
        dummy_coords = np.array([0.0, 0.0, 0.0])
        dummy_nneighs = np.array([0,0,0,0]) # H, C, N, O counts
        try:
            class MinimalAtom:
                def __init__(self, element, element_type, charge, nneighs):
                    self.element = element
                    self.element_type = element_type
                    self.charge = charge
                    self.nneighs = nneighs
            dummy_atom = MinimalAtom(element='C', element_type='C.3', charge=0.0, nneighs=dummy_nneighs)
            self.atom_feature_dim = self._get_atom_features(dummy_atom).shape[0]
        except Exception:
            self.atom_feature_dim = len(self.element_vocab) + len(self.atom_type_vocab) + 1 + 4 # Fallback

    def _get_atom_features(self, atom: Atom) -> torch.Tensor:
        elem_str = atom.element.upper() if isinstance(atom.element, str) else str(atom.element) # Ensure string and uppercase
        element_idx = self.element_to_index.get(elem_str, self.element_to_index['UNK_ELEM'])
        element_one_hot = F.one_hot(torch.tensor(element_idx), num_classes=len(self.element_vocab)).float()

        atom_type_str = str(atom.element_type) # Ensure string
        atom_type_idx = self.atom_type_to_index.get(atom_type_str, self.atom_type_to_index['UNK_ATOM_TYPE'])
        atom_type_one_hot = F.one_hot(torch.tensor(atom_type_idx), num_classes=len(self.atom_type_vocab)).float()

        charge_tensor = torch.tensor([float(atom.charge)], dtype=torch.float32) # Ensure float

        # Ensure nneighs is a tensor of fixed size (e.g., 4 for H, C, N, O)
        nneighs = np.array(atom.nneighs, dtype=float) # Ensure numpy array of floats
        if nneighs.size != 4: # Example fixed size
            # This padding/truncating logic needs to match your `nneighs` definition from `read_mol2`
            corrected_nneighs = np.zeros(4, dtype=float)
            min_len = min(nneighs.size, 4)
            corrected_nneighs[:min_len] = nneighs[:min_len]
            nneighs_tensor = torch.tensor(corrected_nneighs, dtype=torch.float32)
        else:
            nneighs_tensor = torch.tensor(nneighs, dtype=torch.float32)

        return torch.cat([element_one_hot, atom_type_one_hot, charge_tensor, nneighs_tensor])

    def _get_bond_features(self, border_type) -> torch.Tensor:
        # MOL2 bond types can be '1', '2', '3', 'am' (amide), 'ar' (aromatic), 'du' (dummy), 'un' (unknown), 'nc' (not connected)
        bond_type_str = str(border_type).lower()
        bond_idx = self.bond_type_to_index.get(bond_type_str, self.bond_type_to_index['UNK_BOND'])
        return F.one_hot(torch.tensor(bond_idx), num_classes=len(self.bond_type_vocab)).float()

    def featurize_graph(self, ligand: Ligand, center=None, glems=None) -> Data:
        """
        Featurize ligand into GVP-compatible format with proper attribute names
        """
        if not ligand.atoms:
            # Return None for empty ligands
            return None

        # Get basic features
        pos = torch.tensor(ligand.coordinates(), dtype=torch.float32)
        target_pos = pos.clone()  # Keep original coordinates for debugging
        # print first few coords
        if center is not None:
            # move coordinates to center, if center is 0,0,0 and current ligand center is 50,50,50 I want lig center to go to given center
            pos = pos - pos.mean(dim=0)  # Center around origin
            # If center is provided, adjust coordinates
            # Example: if center is [0, 0, 0], pos = pos - pos.mean(dim=0) + center
            # This centers the ligand around the origin
            # If center is a list or tuple, convert to tensor
            pos = pos + center
        if isinstance(glems, dict) and 'atom_single' in glems:
            #the item in dict is tensor. from torch.load(filepath, map_location='cpu') 
            glem_atom_s_feature = glems['atom_single'].clone().detach().float()
            # crop or pad to match ligand.atoms length
            if glem_atom_s_feature.size(0) < len(ligand.atoms):
                padding = torch.zeros(len(ligand.atoms) - glem_atom_s_feature.size(0), glem_atom_s_feature.size(1), dtype=torch.float32)
                x = torch.cat([glem_atom_s_feature, padding], dim=0)
            elif glem_atom_s_feature.size(0) > len(ligand.atoms):
                x = glem_atom_s_feature[:len(ligand.atoms)]
            else:
                x = glem_atom_s_feature
        else: 
            node_features_list = [self._get_atom_features(atom) for atom in ligand.atoms]
            x = torch.stack(node_features_list)
         # Process bonds
        atom_obj_to_idx_map = {atom_obj: i for i, atom_obj in enumerate(ligand.atoms)}

        edge_src, edge_dst, edge_attr_list = [], [], []
        for atom1_obj, atom2_obj, border_type in ligand.bonds:
            idx1 = atom_obj_to_idx_map.get(atom1_obj)
            idx2 = atom_obj_to_idx_map.get(atom2_obj)

            if idx1 is None or idx2 is None:
                print(f"Warning: Atom object in bond not found in ligand's atom list. Skipping bond.")
                continue

            edge_src.extend([idx1, idx2])
            edge_dst.extend([idx2, idx1]) # Add edges in both directions

            bond_f = self._get_bond_features(border_type)
            edge_attr_list.extend([bond_f, bond_f])

        if edge_src:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_attr = torch.stack(edge_attr_list)
        else: # No bonds
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, self.num_bond_features), dtype=torch.float32)

        # Handle empty cases
        if x.size(0) == 0 or edge_index.size(1) == 0:
            return None

        # Create GVP-compatible features
        # Ligands are scalar-only, so we create zero vector features
        node_s = x  # All scalar features [N, feature_dim]
        node_v = torch.zeros(x.size(0), 0, 3, dtype=torch.float32)  # [N, 0, 3] - no vector features

        edge_s = edge_attr  # All scalar edge features [E, feature_dim]
        edge_v = torch.zeros(edge_attr.size(0), 0, 3, dtype=torch.float32)  # [E, 0, 3] - no vector features

        # Debug prints (remove after testing)
        # print(f"Ligand featurization:")
        # print(f"  Nodes: {node_s.shape} scalar, {node_v.shape} vector")
        # print(f"  Edges: {edge_s.shape} scalar, {edge_v.shape} vector")

        return Data(
            # Keep original attributes for backward compatibility
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            target_pos=target_pos,

            # Add GVP-compatible attributes
            node_s=node_s.float(),
            node_v=node_v.float(),
            edge_s=edge_s.float(),
            edge_v=edge_v.float()
        )

# Alternative version that adds vector features from geometry
class LigandFeaturizerWithGeometry(LigandFeaturizer):
    """
    Enhanced ligand featurizer that includes some geometric vector features
    """

    def _compute_bond_vectors(self, ligand: Ligand, pos: torch.Tensor) -> torch.Tensor:
        """
        Compute bond direction vectors for ligand edges
        Returns: [E, 1, 3] tensor of normalized bond directions
        """
        if not ligand.bonds:
            return torch.zeros(0, 1, 3, dtype=torch.float32)

        atom_obj_to_idx_map = {atom_obj: i for i, atom_obj in enumerate(ligand.atoms)}

        bond_vectors = []
        for atom1_obj, atom2_obj, _ in ligand.bonds:
            idx1 = atom_obj_to_idx_map.get(atom1_obj)
            idx2 = atom_obj_to_idx_map.get(atom2_obj)

            if idx1 is not None and idx2 is not None:
                # Bond direction vector
                bond_vec = pos[idx2] - pos[idx1]  # [3]
                bond_vec_normalized = F.normalize(bond_vec.unsqueeze(0), dim=1)  # [1, 3]

                # Add for both directions (undirected graph)
                bond_vectors.extend([bond_vec_normalized, -bond_vec_normalized])

        if bond_vectors:
            vectors = torch.cat(bond_vectors, dim=0)  # [E, 3]
            return vectors.unsqueeze(1)  # [E, 1, 3]
        else:
            return torch.zeros(0, 1, 3, dtype=torch.float32)

    def featurize_graph(self, ligand: Ligand) -> Data:
        """
        Featurize ligand with geometric vector features from bond directions
        """
        if not ligand.atoms:
            return None

        # Get basic featurization from parent class
        data = super().featurize_graph(ligand)
        if data is None:
            return None

        # Add geometric vector features
        pos = data.pos

        # Node vector features: for now, keep as zero (could add atomic orbitals, etc.)
        node_v = torch.zeros(pos.size(0), 0, 3, dtype=torch.float32)

        # Edge vector features: bond directions
        edge_v = self._compute_bond_vectors(ligand, pos)

        # Update the data object
        data.node_v = node_v
        data.edge_v = edge_v

        # print(f"Ligand featurization with geometry:")
        # print(f"  Nodes: {data.node_s.shape} scalar, {node_v.shape} vector")
        # print(f"  Edges: {data.edge_s.shape} scalar, {edge_v.shape} vector")

        return data
