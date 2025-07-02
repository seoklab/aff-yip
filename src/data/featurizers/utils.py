

import torch
import math
import torch.nn.functional as F

import numpy as np
import scipy 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.data.structures import Protein, Ligand, VirtualNode, Residue

from src.data.featurizers.const import AMINOACID

aa_to_idx = {aa: i for i, aa in enumerate(AMINOACID)}

def one_hot_encode_aminoacid(res_name):
    vec = np.zeros(len(AMINOACID), dtype=np.float32)
    idx = aa_to_idx.get(res_name, None)
    if idx is not None:
        vec[idx] = 1.0
    return vec

def aa_one_hot(protein:Protein) -> torch.Tensor:
    """
    Returns a tensor of shape (num_residues, num_amino_acids) with one-hot encoding
    for each residue's amino acid type.
    """
    num_residues = len(protein.residues)
    aa_one_hot = torch.zeros((num_residues, len(AMINOACID)), dtype=torch.float32)

    for i, residue in enumerate(protein.residues):
        if residue.is_ligand or residue.is_water:
            continue  # Skip ligands and water residues
        aa_one_hot[i] = torch.tensor(one_hot_encode_aminoacid(residue.res_name), dtype=torch.float32)

    return aa_one_hot

def water_embeddings(X_water, num_embeddings=16):
    # now just random embeddings
    if X_water.size(0) == 0:
        return torch.empty((0, num_embeddings), dtype=torch.float32)
    return torch.rand(X_water.size(0), num_embeddings, dtype=torch.float32)

def rbf(D, D_min=0, D_max=20, D_count=16, device=None):
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_sigma = (D_max - D_min) / D_count
    return torch.exp(-((D.unsqueeze(-1) - D_mu)**2) / (2 * D_sigma**2))

def positional_embeddings(edge_index, num_embeddings=None):
    d = edge_index[0] - edge_index[1]  # or dst - src depending on convention

    freqs = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def stack_bb_coordinates(protein_obj: Protein) -> torch.Tensor:
    num_residues = len(protein_obj.residues)
    protein_coords_list = []
    residue_s = []
    residue_processed = set()  # To track processed residues
    for residue in protein_obj.residues:
        if str(residue) in residue_processed: 
            continue
        if residue.is_ligand:
            num_residues -= 1 # Skip water or ligand residues
            continue
        n_atom = residue.atom("N")
        ca_atom = residue.atom("CA") # or residue.ca_atom if already populated
        c_atom = residue.atom("C")

        if not (n_atom and ca_atom and c_atom):
            raise ValueError(
                f"Residue {residue} in protein {protein_obj.name} "
                "is missing one or more backbone atoms (N, CA, C) required for dihedral calculation."
            )
        residue_s.append(residue)
        residue_processed.add(str(residue))  # Mark this residue as processed

        protein_coords_list.append(torch.tensor(n_atom.coordinates, dtype=torch.float32))
        protein_coords_list.append(torch.tensor(ca_atom.coordinates, dtype=torch.float32))
        protein_coords_list.append(torch.tensor(c_atom.coordinates, dtype=torch.float32))
    # print (f'number of residues in protein {protein_obj.name}: {num_residues}, {len(residue_s)} residues with N, CA, C atoms')
    X_protein_flat = torch.stack(protein_coords_list)
    return X_protein_flat, residue_s


def stack_water_coordinates(protein_obj: Protein) -> torch.Tensor:
    """
    Stacks water coordinates from the protein object into a tensor.
    Returns a tensor of shape (num_water_molecules, 3) where each water molecule
    is represented by Oxygen atom coordinates.
    """
    water_coords_list = []
    for residue in protein_obj.residues:
        if not residue.is_water:
            continue
        o_atom = residue.atom("O")

        water_coords_list.append(torch.tensor(o_atom.coordinates, dtype=torch.float32))
    if len(water_coords_list) == 0:
        return torch.empty((0, 3), dtype=torch.float32)  # Return empty tensor if no water found
    X_water_flat = torch.stack(water_coords_list)

    return X_water_flat 


def residue_dihedrals(X, eps: float = 1e-7) -> torch.Tensor: 
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

def _normalize_torch(vector: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalizes a `torch.Tensor` along dimension `dim` without `nan`s."""
    
    return torch.nan_to_num(
        torch.div(vector, torch.norm(vector, dim=dim, keepdim=True)))

def sidechain_orientation(X) -> np.ndarray:
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
    vec = torch.tensor(vec, dtype=torch.float32)
    return vec


def backbone_orientation(X):
    X_ca = X[1::3, :] # Shape: (num_residues, 3)
    
    # --- Calculate vectors pointing FORWARD: C_alpha(i) -> C_alpha(i+1) ---
    # Differences: C_alpha[i+1] - C_alpha[i]
    forward_diffs = X_ca[1:] - X_ca[:-1]  # Shape: (num_residues - 1, 3)
    norm_forward_vectors = F.normalize(forward_diffs, p=2, dim=-1, eps=1e-8)

    # Pad with a zero vector at the END (for the last residue's forward vector)
    # Input shape (N-1, 3), pad tuple (pad_left, pad_right, pad_top, pad_bottom) for 2D
    # (0,0) for last dim (coords), (0,1) for first dim (residues)
    forward_vectors_padded = F.pad(norm_forward_vectors, (0, 0, 0, 1), mode='constant', value=0.0) # (num_residues, 3)

    # --- Calculate vectors pointing BACKWARD: C_alpha(i) -> C_alpha(i-1) ---
    # Differences: C_alpha[i-1] - C_alpha[i] (Note: X_ca[:-1] are C_alpha_0 to C_alpha_N-2 etc.)
    # backward_diffs[k] is C_alpha[k] - C_alpha[k+1]
    backward_diffs = X_ca[:-1] - X_ca[1:] # Shape: (num_residues - 1, 3)
    norm_backward_vectors = F.normalize(backward_diffs, p=2, dim=-1, eps=1e-8)
    # Pad with a zero vector at the BEGINNING (for the first residue's backward vector)
    backward_vectors_padded = F.pad(norm_backward_vectors, (0, 0, 1, 0), mode='constant', value=0.0) # (num_residues, 3)

    # --- Combine into the final orientation tensor ---
    # Unsqueeze to prepare for concatenation: (num_residues, 3) -> (num_residues, 1, 3)
    forward_final = forward_vectors_padded.unsqueeze(-2)
    backward_final = backward_vectors_padded.unsqueeze(-2)

    # Concatenate along the new dimension (dim 1, or -2)
    # result[i, 0, :] will be the forward vector for residue i
    # result[i, 1, :] will be the backward vector for residue i
    orientation_features = torch.cat([forward_final, backward_final], dim=-2) # (num_residues, 2, 3)
    return orientation_features


class GridOption:
    def __init__(self,padding,gridsize,option,clash,shellsize):
        self.padding = padding
        self.gridsize = gridsize
        self.option = option
        self.clash = clash
        self.shellsize= shellsize # throw if no contact within this distance


def pocket_virtual_nodes(receptor:Protein,ligand:Ligand,
                        only_backbone=False,
                        opt=None,
                        gridout=None):
    if opt is None:
        opt = GridOption(padding=5.0, gridsize=1.0, option='ligand', clash=1.8, shellsize=6.0)
    if only_backbone: 
        # xyzs_rec = stack_residue_coordinates(receptor)
        xyzs_rec = receptor.ncaco_coordinates() 
    else: 
        xyzs_rec = receptor.all_coordinates()

    xyzs_lig = ligand.coordinates()

    if opt.option == 'ligand':
        bmin = np.min(xyzs_lig[:,:]-opt.padding,axis=0)
        bmax = np.max(xyzs_lig[:,:]+opt.padding,axis=0)
    elif opt.option == 'basic':
        bmin = np.min(xyzs_rec[:,:]-opt.padding,axis=0)
        bmax = np.max(xyzs_rec[:,:]+opt.padding,axis=0)

    imin = [int(bmin[k]/opt.gridsize)-1 for k in range(3)]
    imax = [int(bmax[k]/opt.gridsize)+1 for k in range(3)]

    grids = []
    # print("detected %d grid points..."%((imax[0]-imin[0])*(imax[1]-imin[1])*(imax[2]-imin[2])))
    for ix in range(imin[0],imax[0]+1):
        for iy in range(imin[1],imax[1]+1):
            for iz in range(imin[2],imax[2]+1):
                grid = np.array([ix*opt.gridsize,iy*opt.gridsize,iz*opt.gridsize])
                grids.append(grid)

    grids = np.array(grids)

    # Remove clashing or far-off grids
    kd      = scipy.spatial.cKDTree(grids)
    kd_ca   = scipy.spatial.cKDTree(xyzs_rec)
    excl = np.concatenate(kd_ca.query_ball_tree(kd, opt.clash)) #clashing
    incl = np.unique(np.concatenate(kd_ca.query_ball_tree(kd, opt.shellsize))) #grid-rec shell

    if opt.option == 'ligand': 
        kd_lig  = scipy.spatial.cKDTree(xyzs_lig)
        ilig = np.unique(np.concatenate(kd_lig.query_ball_tree(kd, opt.padding))) # ligand environ
        interface = np.unique(np.array([i for i in incl if (i not in excl and i in ilig)],dtype=np.int16))
        grids = grids[interface]
    
    elif opt.option == 'basic':
        interface = np.unique(np.array([i for i in incl if (i not in excl)],dtype=np.int16))
        grids = grids[interface]

    D = scipy.spatial.distance_matrix(grids,grids)
    graph = csr_matrix((D<(opt.gridsize+0.1)).astype(int))
    n, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    ncl = [sum(labels==k) for k in range(n)]
    biggest = np.unique(np.where(labels==np.argmax(ncl)))
    grids = grids[biggest]

    virtual_nodes = [VirtualNode(coord) for coord in grids]

    if gridout is not None:
        for i,grid in enumerate(grids):
            gridout.write("HETATM %4d  CA  CA  X   1    %8.3f%8.3f%8.3f\n"%(i,grid[0],grid[1],grid[2]))
    return virtual_nodes


if __name__ == '__main__':
    # Example usage
    mol2_file = '/home/j2ho/DB/pdbbind/v2020-refined/5hz8/5hz8_ligand.mol2'
    pdb_file = '5hz8.pdb'

    protein = Protein(pdb_filepath=pdb_file, read_water=True, read_ligand=False)
    ligand = Ligand(mol2_filepath=mol2_file)
