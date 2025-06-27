import numpy as np

from src.io.protein_parser import read_pdb
from src.io.ligand_parser import read_mol2


class Atom:
    def __init__(self, name, element, coordinates, charge=None, b_factor=None, element_type=None, nneighs=None):
        # Common attributes
        self.name = name # e.g., 'CA', 'CB' in PDB, 'C1', 'C2' in mol2
        self.element = element # e.g., 'C', 'N', 'O', 'H', 'S', 'P'
        self.coordinates = np.array(coordinates)
        self.charge = charge
        self.is_metal = self.check_metal(element)  # True if it's a metal that we want to consider !!! 

        # PDB file only attributes
        self.b_factor = b_factor 

        # MOL2 file only attributes
        self.element_type = element_type  # e.g., C.3, C.ar in mol2 files (for ligand)
        self.nneighs = nneighs  # Number of neighboring atoms of each type [H,C,N,O] for each atom (for ligand)

    def check_metal(self, element):
        """Check if the element is a metal."""
        metals = ['CA', 'ZN', 'MN', 'MG', 'FE', 'CD', 'CO', 'CU']
        return element in metals
    
    def __repr__(self):
        return f"{self.name}"

    
class Residue:
    def __init__(self, res_name, res_num, chain_id):
        self.res_name = res_name
        self.chain_id = chain_id
        self.res_num = res_num
        self.atoms = [] # List of Atom objects
        self.ca_atom = None
        self.is_water = (res_name in ['HOH','DOD'])  # Commonly used name for water in PDB files
        self.is_ligand = False  # Flag for ligand residues, set to True if this residue is a ligand

    def add_atom(self, atom):
        self.atoms.append(atom)
        if atom.name == 'CA':
            self.ca_atom = atom

    def get_atom(self, atom_name_query): 
        for atom in self.atoms:
            if atom.name == atom_name_query:
                return atom
        return None

    def get_ca(self):
        for atom in self.atoms:
            if atom.name == 'CA':
                self.ca_atom = atom
                return atom
        return None
    
    def get_cb(self):
        for atom in self.atoms:
            if atom.name == 'CB':
                return atom
        return None

    def get_coordinates(self):
        """Get the coordinates of all atoms in the residue."""
        return np.array([atom.coordinates for atom in self.atoms])

    def __repr__(self):
        return f"({self.res_name}.{self.chain_id}.{self.res_num})"


class Protein:
    def __init__(self, pdb_filepath=None, read_water=False, read_ligand=False, excl_aa_types=None, excl_chain=None, read_chain=None):
        self.pdb_filepath = pdb_filepath
        self.name = pdb_filepath.split('/')[-1].split('.')[0] if pdb_filepath else None
        self.residues = []

        # Optional parameters for filtering residues
        self.read_water = read_water
        self.read_ligand = read_ligand
        self.excl_aa_types = excl_aa_types if excl_aa_types is not None else []
        self.excl_chain = excl_chain if excl_chain is not None else []
        self.read_chain = read_chain if read_chain is not None else []  # Only read these chains, if specified

        if pdb_filepath:
            self._load_pdb()

    def _load_pdb(self):
        # ... logic to create Residue and Atom objects and link them ...
        res_names, res_chains, xyz, atms, elems, lig_flag = read_pdb(self.pdb_filepath, 
                                                      read_ligand=self.read_ligand, 
                                                      read_water=self.read_water, 
                                                      excl_aa_types=self.excl_aa_types, 
                                                      excl_chain=self.excl_chain,
                                                      read_chain=self.read_chain)
        self.res_name_list = res_names
        self.res_chain_id_list = res_chains

        for resname, reschain in zip(res_names, res_chains):
            parts = reschain.split('.')
            chain_id = parts[0]  # e.g., 'A.72' -> chain_id = 'A'
            try:
                res_num = int(parts[1])
            except ValueError:
                # Handle cases like insertion codes, e.g., 'A.10A'
                res_num_str = ""
                for char_ in parts[1]:
                    if char_.isdigit() or char_ == '-': # allow negative numbers
                        res_num_str += char_
                    else: # Stop at first non-digit (e.g. insertion code)
                        break 
                if not res_num_str: # if nothing was found, it is not a valid number.
                    print(f"Warning: Could not parse residue number from '{parts[1]}' in '{reschain_id_str}'. Skipping residue.")
                    continue
                res_num = int(res_num_str)

            residue = Residue(res_name=resname, res_num=res_num, chain_id=chain_id)
            if reschain not in xyz:
                print(f"Warning: Residue '{reschain}' not found in xyz data. Skipping residue.")
            for atm_name, coords in xyz[reschain].items():  # atm_name: e.g., 'CA', 'CB', 'C1', 'C2'
                element = elems[reschain][atm_name] if atm_name in elems[reschain] else 'X'  # Default to 'X' if not found
                atm = Atom(atm_name, element=element, coordinates=coords)
                residue.add_atom(atm)
            if lig_flag[reschain]:
                residue.is_ligand = True
            self.residues.append(residue)

    def get_coordinates(self):
        """Get the coordinates of all atoms in the protein."""
        xyz = [np.array(self.residues[i].get_coordinates(), dtype=np.float32) for i in range(len(self.residues))]
        xyz = np.concatenate(xyz)
        return xyz
    
    def get_ncaco_coordinates(self):
        """Get the coordinates of N, CA, C, O atoms in the protein."""
        ncaco_coords = []
        for residue in self.residues:
            if residue.is_water or (self.excl_aa_types and residue.res_name in self.excl_aa_types):
                continue
            n_atom = residue.get_atom('N')
            ca_atom = residue.get_ca()
            c_atom = residue.get_atom('C')
            o_atom = residue.get_atom('O')
            for atom in [n_atom, ca_atom, c_atom, o_atom]:
                if atom is not None:
                    ncaco_coords.append(atom.coordinates)

        return np.stack(ncaco_coords)
    
    def get_sidechain_info(self): 
        sidechain_info = {} 
        for residue in self.residues:
            sidechain_coords = {} 
            if residue.is_water or (self.excl_aa_types and residue.res_name in self.excl_aa_types):
                continue
            for atom in residue.atoms:
                if atom.name in ['N', 'CA', 'C', 'O']:
                    continue
                sidechain_coords[atom.name] = atom.coordinates
            sidechain_info[str(residue)] = sidechain_coords
        return sidechain_info
 
    def get_water_coordinates(self):
        """Get the coordinates of water molecules in the protein."""
        water_coords = []
        for residue in self.residues:
            if residue.is_water:
                for atom in residue.atoms:
                    water_coords.append(atom.coordinates)
        return np.array(water_coords)

class Ligand:
    def __init__(self, mol2_filepath=None, drop_H=False):
        self.mol2_filepath = mol2_filepath
        self.atoms = []
        self.bonds = [] # List of tuples (Atom1, Atom2, border)
        self.drop_H = drop_H
        if mol2_filepath:
            self._load_mol2()

    def _load_mol2(self):
        # Logic to read the MOL2 file and create Atom objects
        elems, qs, bonds, borders, xyzs, nneighs, atms, atypes = read_mol2(self.mol2_filepath, drop_H=self.drop_H)
        # nneighs: number of neighboring atoms of each type [H,C,N,O] for each atom
        # e.g. if an atom has 2 C neighbors, 1 N neighbor, and 0 H and O neighbors, it will be [0,2,1,0]
        for elem, q, xyz, atm, atype, nneighs in zip(elems, qs, xyzs, atms, atypes, nneighs):
            atom = Atom(name=atm, element=elem, coordinates=xyz, charge=q, element_type=atype, nneighs=nneighs)
            self.atoms.append(atom)

        for bond, border in zip(bonds, borders):
            atom1 = self.atoms[bond[0]]
            atom2 = self.atoms[bond[1]]
            self.bonds.append((atom1, atom2, border))
    
    def get_coordinates(self):
        """Get the coordinates of all atoms in the ligand."""
        return np.array([atom.coordinates for atom in self.atoms])
    
    def get_atom_elements(self):
        """Get a list of all atoms in the ligand."""
        return np.array([atom.element for atom in self.atoms])

class VirtualNode:
    def __init__(self, coordinates, node_type='virtual'):
        self.coordinates = np.array(coordinates)
        self.node_type = node_type  # e.g., 'virtual', 'pocket', etc.
        # self.water_occupancy = None

    def set_water_occupancy(self, water_coords, sigma=2.0, cutoff_distance=6.0, 
                          normalization='none'):
        """
        Calculate water occupancy with various normalization options.
        
        Parameters:
        -----------
        water_coords : np.array or torch.Tensor
            Array of water coordinates, shape (n_waters, 3)  
        sigma : float, default=2.0
            Standard deviation for Gaussian distribution
        cutoff_distance : float, default=6.0
            Maximum distance to consider water molecules
        normalization : str, default='none'
            'none': Raw sum of Gaussian weights
            'count': Divide by number of water molecules within cutoff
            'max_possible': Divide by theoretical maximum (all waters at node position)
            'density': Normalize by local volume
        
        Returns:
        --------
        float : occupancy value
        """
        if hasattr(water_coords, 'detach'):  # Check if it's a torch tensor
            water_coords = water_coords.detach().cpu().numpy()
        elif not isinstance(water_coords, np.ndarray):
            water_coords = np.array(water_coords)

        if len(water_coords) == 0:
            self.water_occupancy = 0.0
            return self.water_occupancy
        
        # Calculate distances
        distances = np.linalg.norm(water_coords - self.coordinates, axis=1)
        
        # Apply cutoff
        mask = distances <= cutoff_distance
        if not np.any(mask):
            self.water_occupancy = 0.0
            return self.water_occupancy
        
        relevant_distances = distances[mask]
        n_relevant_waters = len(relevant_distances)
        
        # Calculate Gaussian weights
        weights = np.exp(-0.5 * (relevant_distances / sigma) ** 2)
        raw_occupancy = np.sum(weights)
        
        # Apply normalization
        if normalization == 'none':
            self.water_occupancy = raw_occupancy
        elif normalization == 'count':
            self.water_occupancy = raw_occupancy / n_relevant_waters
        elif normalization == 'max_possible':
            # Maximum possible weight (if all waters were at the node position)
            max_weight = n_relevant_waters * 1.0  # Gaussian peak value is 1
            self.water_occupancy = raw_occupancy / max_weight
        elif normalization == 'density':
            # Normalize by the volume of the sphere within cutoff
            volume = (4/3) * np.pi * cutoff_distance**3
            self.water_occupancy = raw_occupancy / volume
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
        
        return self.water_occupancy

    def __repr__(self):
        return f"VirtualNode(type={self.node_type}, coords={self.coordinates})"
    

if __name__ == '__main__':
    # Example usage
    pdb_file = '5hz8.pdb'
    mol2_file = '/home/j2ho/DB/pdbbind/v2020-refined/5hz8/5hz8_ligand.mol2'
    protein = Protein(pdb_filepath=pdb_file, read_water=True)
    ligand = Ligand(mol2_filepath=mol2_file)
    xyzs = protein.get_coordinates()
    print ("Protein Coordinates:", xyzs)