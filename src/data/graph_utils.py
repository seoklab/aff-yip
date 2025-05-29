
import os
import sys
import random
import torch
import json
import tqdm, random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster

from pathlib import Path
import numpy as np

from structures import Atom, Residue, Protein, Ligand, VirtualNode
from parse_utils import generate_virtual_nodes, featurize_virtual_nodes
import const 

def lig_graph_gen(mol2file):
    pass 

def receptor_graph_gen_AA(protein: Protein, virtual_nodes: list[VirtualNode] = None):
    residues = protein.residues
    print (residues)
    res_coords = torch.as_tensor([atom.coordinates for res in residues for atom in res.atoms], dtype=torch.float32)
    print (coords)
    G = None 
    return G

def receptor_graph_gen(protein: Protein, virtual_nodes: list[VirtualNode] = None):
    """
    Generate a residue level receptor graph from a Protein object.
    
    Parameters:
    - protein: Protein object containing the receptor structure.
    - virtual_nodes: List of VirtualNode objects (optional).
    
    Returns:
    - G: Graph representation of the receptor.
    """
    residues = protein.residues
    for residue in residues: 
        atms = residue.atoms
        for atm in atms: 
            print(residue, atm.name, atm.coordinates)
        sys.exit() 
    return G


def stack_residue_coordinates(protein_obj: Protein) -> torch.Tensor:
    num_residues = len(protein_obj.residues)
    protein_coords_list = []
    for residue in protein_obj.residues:
        if residue.is_ligand:
            num_residues -= 1 # Skip water or ligand residues
            continue
        n_atom = residue.get_atom("N")
        ca_atom = residue.get_atom("CA") # or residue.ca_atom if already populated
        c_atom = residue.get_atom("C")

        if not (n_atom and ca_atom and c_atom):
            raise ValueError(
                f"Residue {residue} in protein {protein_obj.name} "
                "is missing one or more backbone atoms (N, CA, C) required for dihedral calculation."
            )

        protein_coords_list.append(torch.tensor(n_atom.coordinates, dtype=torch.float32))
        protein_coords_list.append(torch.tensor(ca_atom.coordinates, dtype=torch.float32))
        protein_coords_list.append(torch.tensor(c_atom.coordinates, dtype=torch.float32))
    
    X_protein_flat = torch.stack(protein_coords_list)

    return X_protein_flat 

def get_residue_dihedrals(X, eps: float = 1e-7) -> torch.Tensor: 
    # Originally from https://github.com/jingraham/neurips19-graph-protein-design
    # Adapted version from https://github.com/drorlab/gvp
    """
    Returns:
        torch.Tensor: A tensor of shape (num_residues, 6) representing
                      [cos(phi), sin(phi), cos(psi), sin(psi), cos(omega), sin(omega)]
                      for each residue.
    """
    X_protein_flat = X
    dX = X_protein_flat[1:, :] - X_protein_flat[:-1, :] # (num_residues * 3 - 1, 3)
    if X.shape[0] < 2:
        D = torch.zeros((X.shape[0], 3), device=X_protein_flat.device, dtype=X_protein_flat.dtype)
    else:
        U = F.normalize(dX, dim=-1)
        u_2 = U[:-2, :]
        u_1 = U[1:-1, :]
        u_0 = U[2:, :]

        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        
        D_raw = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD) # (num_residues * 3 - 3)

        D = F.pad(D_raw, (1, 2), 'constant', 0) # remove phi[0] and psi[-1], omega[-1] and pad with zeros
        # D = D.view((X, 3)) # Reshape to (num_residues, 3_angles)
        D = torch.reshape(D, (-1, 3)) # Reshape to (num_residues, 3_angles)
    D_features = torch.cat((torch.cos(D), torch.sin(D)), dim=-1) # convert angles to cos/sin pairs, (num_residues, 6)
    return D_features

def _normalize_np(vector: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Helper function to normalize a NumPy vector, handling zero vectors."""
    norm = np.linalg.norm(vector)
    if norm < epsilon:
        return np.zeros_like(vector, dtype=np.float32)
    return vector.astype(np.float32) / norm

def get_sidechain_orientation(X) -> np.ndarray:
    """
    Calculates an idealized sidechain orientation vector from backbone N, CA, C coordinates.
    The formula is `vec = -bisector * sqrt(1/3) - perp * sqrt(2/3)`, often used
    in constructing local reference frames (e.g., in AlphaFold-related architectures).

    Returns:
        np.ndarray: A normalized 3D vector representing the idealized sidechain orientation.
                    Returns a zero vector if N, CA, C are collinear or form a zero vector.
    """
    X_np = X.numpy()
    num_residues = X_np.shape[0] // 3
    X_reshaped = X_np.reshape(num_residues, 3, 3)

    n_coord = X_reshaped[:, 0, :]  # N coords for all residues
    ca_coord = X_reshaped[:, 1, :] # CA coords for all residues
    c_coord = X_reshaped[:, 2, :]  # C coords for all residues

    n_coord = np.asarray(n_coord, dtype=np.float32)
    ca_coord = np.asarray(ca_coord, dtype=np.float32)
    c_coord = np.asarray(c_coord, dtype=np.float32)

    # Vectors from CA to N and C
    ca_to_n = _normalize_np(n_coord - ca_coord)
    ca_to_c = _normalize_np(c_coord - ca_coord)

    # If N, CA, C are collinear or CA is at the same position as N or C,
    # ca_to_n or ca_to_c might be zero vectors. _normalize_np handles this.
    # If both are zero (e.g. all three points identical), bisector and perp will be zero.

    bisector = _normalize_np(ca_to_c + ca_to_n)
    # The cross product order (c, n) matches your reference _sidechains(X)
    # where X[:,0]=N, X[:,1]=CA (origin), X[:,2]=C.
    # So, c = X[:,2]-X[:,1] (C-CA), n = X[:,0]-X[:,1] (N-CA).
    # perp = cross(C-CA, N-CA)
    perp = _normalize_np(np.cross(ca_to_c, ca_to_n))

    # If ca_to_c and ca_to_n are collinear (e.g. angle is 0 or 180), perp will be zero.
    # If either ca_to_c or ca_to_n is zero, perp will be zero.
    # If bisector and perp are both zero vectors, vec will be zero.

    vec = -bisector * math.sqrt(1.0/3.0) - perp * math.sqrt(2.0/3.0)
    
    # The resulting vector 'vec' is not necessarily normalized by the formula itself,
    # though its components (bisector, perp) are.
    # The reference _sidechains does not show a final normalization of 'vec'.
    # Depending on usage, it might need to be normalized.
    # For now, returning as per formula. If a unit vector is always needed:
    vec = _normalize_np(vec) 
    return vec

def get_backbone_orientation(X):
    X_ca = X[1::3, :] # Shape: (num_residues, 3)
    
    # --- Calculate vectors pointing FORWARD: C_alpha(i) -> C_alpha(i+1) ---
    # Differences: C_alpha[i+1] - C_alpha[i]
    # This results in (num_residues - 1) vectors.
    forward_diffs = X_ca[1:] - X_ca[:-1]  # Shape: (num_residues - 1, 3)
    norm_forward_vectors = F.normalize(forward_diffs, p=2, dim=-1, eps=1e-8)

    # Pad with a zero vector at the END (for the last residue's forward vector)
    # Input shape (N-1, 3), pad tuple (pad_left, pad_right, pad_top, pad_bottom) for 2D
    # (0,0) for last dim (coords), (0,1) for first dim (residues)
    forward_vectors_padded = F.pad(norm_forward_vectors, (0, 0, 0, 1), mode='constant', value=0.0)
    # Shape: (num_residues, 3)

    # --- Calculate vectors pointing BACKWARD: C_alpha(i) -> C_alpha(i-1) ---
    # Differences: C_alpha[i-1] - C_alpha[i] (Note: X_ca[:-1] are C_alpha_0 to C_alpha_N-2 etc.)
    # To get C_alpha[i-1] - C_alpha[i], we can use X_ca[:-1] (as C_alpha[i-1]) and X_ca[1:] (as C_alpha[i])
    # This also results in (num_residues - 1) vectors.
    # backward_diffs[k] is C_alpha[k] - C_alpha[k+1]
    backward_diffs = X_ca[:-1] - X_ca[1:] # Shape: (num_residues - 1, 3)
    norm_backward_vectors = F.normalize(backward_diffs, p=2, dim=-1, eps=1e-8)
    # Pad with a zero vector at the BEGINNING (for the first residue's backward vector)
    # (0,0) for last dim (coords), (1,0) for first dim (residues)
    backward_vectors_padded = F.pad(norm_backward_vectors, (0, 0, 1, 0), mode='constant', value=0.0)
    # Shape: (num_residues, 3)

    # --- Combine into the final orientation tensor ---
    # Unsqueeze to prepare for concatenation: (num_residues, 3) -> (num_residues, 1, 3)
    forward_final = forward_vectors_padded.unsqueeze(-2)
    backward_final = backward_vectors_padded.unsqueeze(-2)

    # Concatenate along the new dimension (dim 1, or -2)
    # result[i, 0, :] will be the forward vector for residue i
    # result[i, 1, :] will be the backward vector for residue i
    orientation_features = torch.cat([forward_final, backward_final], dim=-2)
    # Shape: (num_residues, 2, 3)

    return orientation_features

if __name__ == '__main__':
    # Example usage
    mol2_file = '/home/j2ho/DB/pdbbind/v2020-refined/5hz8/5hz8_ligand.mol2'
    pdb_file = '5hz8.pdb'

    protein = Protein(pdb_filepath=pdb_file, read_water=True, read_ligand=False)
    ligand = Ligand(mol2_filepath=mol2_file)

    X = stack_residue_coordinates(protein)
    D = get_residue_dihedrals(X)
    sidechain_orientation = get_sidechain_orientation(X)
    backbone_orientation = get_backbone_orientation(X)
    # virtual_nodes = generate_virtual_nodes
    # featurize_virtual_nodes(virtual_nodes)  # Define your virtual nodes if needed
    