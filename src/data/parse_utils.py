# pdb and mol2 file reading utilities
# grid generation utilities

import numpy as np
import sys
import const 
from pathlib import Path
import scipy 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def read_pdb(pdb:Path, read_ligand=False, read_water=False,
             excl_aa_types=[], excl_chain=[]):
    resnames = []
    reschains = []
    lig_flag = {} 
    xyz = {}
    atms = {}
    elems = {}

    for l in open(pdb):
        is_lig = False
        if read_ligand or read_water: 
            if l.startswith('HETATM'): 
                is_lig = True
            if not (l.startswith('ATOM') or l.startswith('HETATM')): continue
        else:
            if not l.startswith('ATOM'): continue
        if l.startswith('ENDMDL'): break
        atm = l[12:17].strip()
        aa3 = l[17:20].strip()
        if len(l) >= 76:
            elem = l[76:].strip()
        else: 
            elem = None 
        if not read_ligand and read_water:
            if l.startswith('HETATM'): 
                if not aa3 in ['HOH', 'DOD']: 
                    continue

        if excl_aa_types != [] and aa3 in excl_aa_types: continue
        if l[21] in excl_chain: continue

        reschain = l[21]+'.'+l[22:27].strip() # resnum sometimes can be negative. e.g.) 'A.-1' 
        # (e.g.) atm: OD1 / aa3: ASP / reschain A.72

        if aa3[:2] in const.METAL: aa3 = aa3[:2]
        if aa3 in const.AMINOACID:
            if atm == 'CA':
                resnames.append(aa3)
                reschains.append(reschain)
        elif aa3 in const.NUCLEICACID:
            if atm == "C1'":
                resnames.append(aa3)
                reschains.append(reschain)
        elif aa3 in const.METAL:
            resnames.append(aa3)
            reschains.append(reschain)
        elif read_ligand and reschain not in reschains:
            resnames.append(aa3)
            reschains.append(reschain)
        elif read_water and reschain not in reschains: 
            resnames.append(aa3)
            reschains.append(reschain)

        if reschain not in xyz:
            xyz[reschain] = {}
            atms[reschain] = []
            elems[reschain] = {}
        
        xyz[reschain][atm] = np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
        atms[reschain].append(atm)
        elems[reschain][atm] = elem if elem is not None else 'X'  # Default to 'X' if no element info
        lig_flag[reschain] = is_lig
    return resnames, reschains, xyz, atms, elems, lig_flag


def read_mol2(mol2:Path, drop_H=False):
    read_cont = 0
    qs = []
    elems = []
    xyzs = []
    bonds = []
    borders = []
    atms = []
    atypes = []

    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            continue
        if l.startswith('@<TRIPOS>BOND'):
            read_cont = 2
            continue
        if l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = -1
            continue
        if l.startswith('@<TRIPOS>SUBSTRUCTURE'):
            break

        words = l[:-1].split()
        if read_cont == 1:

            idx = words[0]
            if words[1].startswith('BR'): words[1] = 'Br'
            if words[1].startswith('Br') or  words[1].startswith('Cl') :
                elem = words[1][:2]
            else:
                elem = words[1][0]

            if elem == 'A' or elem == 'B' :
                elem = words[5].split('.')[0]

            if elem not in const.ELEMS:
                elem = 'X'

            atms.append(words[1])
            atypes.append(words[5])
            elems.append(elem)
            qs.append(float(words[-1]))
            xyzs.append([float(words[2]),float(words[3]),float(words[4])])

        elif read_cont == 2:
            bonds.append([int(words[1])-1,int(words[2])-1]) #make 0-index
            bondtypes = {'1':1,'2':2,'3':3,'ar':3,'am':2, 'du':0, 'un':0}
            borders.append(bondtypes[words[3]])

    # nneighs: number of neighboring atoms of each type [H,C,N,O] for each atom
    # e.g. if an atom has 2 C neighbors, 1 N neighbor, and 0 H and O neighbors, it will be [0,2,1,0]
    nneighs = [[0,0,0,0] for _ in qs]
    for i,j in bonds:
        if elems[i] in ['H','C','N','O']:
            k = ['H','C','N','O'].index(elems[i])
            nneighs[j][k] += 1.0
        if elems[j] in ['H','C','N','O']:
            l = ['H','C','N','O'].index(elems[j])
            nneighs[i][l] += 1.0

    # drop hydrogens
    if drop_H:
        nonHid = [i for i,a in enumerate(elems) if a != 'H']
    else:
        nonHid = [i for i,a in enumerate(elems)]

    bonds = [[i,j] for i,j in bonds if i in nonHid and j in nonHid]
    borders = [b for b,ij in zip(borders,bonds) if ij[0] in nonHid and ij[1] in nonHid] # bond types (bond order)
    # atypes = C.3, C.ar, ... 
    # atms = 'N1', 'C2', 'C3',... 
    return np.array(elems)[nonHid], np.array(qs)[nonHid], bonds, borders, np.array(xyzs)[nonHid], np.array(nneighs)[nonHid], atms, np.array(atypes)


class GridOption:
    def __init__(self,padding,gridsize,option,clash):
        self.padding = padding
        self.gridsize = gridsize
        self.option = option
        self.clash = clash
        self.shellsize=7.0 # throw if no contact within this distance


def generate_virtual_nodes(xyzs_rec,xyzs_lig,
                  opt=None,
                  gridout=None):
    if opt is None:
        opt = GridOption(padding=10.0, gridsize=1.0, option='ligand', clash=1.8)
    
    if opt.option == 'ligand':
        bmin = np.min(xyzs_lig[:,:]-opt.padding,axis=0)
        bmax = np.max(xyzs_lig[:,:]+opt.padding,axis=0)
    elif opt.option == 'basic':
        bmin = np.min(xyzs_rec[:,:]-opt.padding,axis=0)
        bmax = np.max(xyzs_rec[:,:]+opt.padding,axis=0)

    imin = [int(bmin[k]/opt.gridsize)-1 for k in range(3)]
    imax = [int(bmax[k]/opt.gridsize)+1 for k in range(3)]

    grids = []
    print("detected %d grid points..."%((imax[0]-imin[0])*(imax[1]-imin[1])*(imax[2]-imin[2])))
    for ix in range(imin[0],imax[0]+1):
        for iy in range(imin[1],imax[1]+1):
            for iz in range(imin[2],imax[2]+1):
                grid = np.array([ix*opt.gridsize,iy*opt.gridsize,iz*opt.gridsize])
                grids.append(grid)

    grids = np.array(grids)
    nfull = len(grids)

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

    n1 = len(grids)
    D = scipy.spatial.distance_matrix(grids,grids)
    graph = csr_matrix((D<(opt.gridsize+0.1)).astype(int))
    n, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    ncl = [sum(labels==k) for k in range(n)]
    biggest = np.unique(np.where(labels==np.argmax(ncl)))
    
    grids = grids[biggest]

    print("Search through %d grid points, of %d contact grids %d clash -> %d, remove outlier -> %d"%(nfull,len(incl),len(excl),n1,len(grids)))

    if gridout is not None:
        for i,grid in enumerate(grids):
            gridout.write("HETATM %4d  CA  CA  X   1    %8.3f%8.3f%8.3f\n"%(i,grid[0],grid[1],grid[2]))
    return grids


def featurize_virtual_nodes(protein_xyz, ligand_xyz, grid_generation_params, 
                            occupancy_data_path=None, dipole_data_path=None):
    """
    Featurize virtual nodes based on protein and ligand coordinates.
    Inputs:
        protein_xyz: np.ndarray
            Coordinates of protein atoms (shape: [N, 3]).
        ligand_xyz: np.ndarray
            Coordinates of ligand atoms (shape: [M, 3]).
        grid_generation_params: GridOption
            Parameters for grid generation.
        occupancy_data_path: str, optional
            Path to occupancy data file.
        dipole_data_path: str, optional 
            Path to dipole data file.
    
    Returns:
        protein_dict: dict
            Dictionary containing protein features.
        ligand_dict: dict
            Dictionary containing ligand features.
        virtual_node_dict: dict
            Dictionary containing virtual node features.            
    """
    protein_dict = {} 
    ligand_dict = {} 
    virtual_node_dict = {}  
    return protein_dict, ligand_dict, virtual_node_dict


if __name__ == '__main__':
    # Example usage
    pdb_file = '/home/j2ho/DB/pdbbind/v2020-refined/5hz8/5hz8_protein.pdb'
    mol2_file = '/home/j2ho/DB/pdbbind/v2020-refined/5hz8/5hz8_ligand.mol2'
    pdb_file = '5hz8.pdb'
    rec_resnames, rec_reschains, rec_xyzs, rec_atms, rec_elems, lig_flag = read_pdb(pdb_file, read_ligand=True, read_water=True)
    print("Protein Residues:", rec_resnames)
    print("Protein Chains:", rec_reschains)
    print("Protein Coordinates:", rec_xyzs)
    print("Protein Atoms:", rec_atms)
    print ("Protein Elements:", rec_elems)
    
    lig_elems, lig_qs, lig_bonds, lig_borders, lig_xyzs, lig_nneighs, lig_atms, lig_atypes = read_mol2(mol2_file)
    # print("MOL2 Elements:", lig_xyzs)
    print (lig_elems)
    # protein_dict, ligand_dict, virtual_node_dict = featurize_virtual_nodes(
