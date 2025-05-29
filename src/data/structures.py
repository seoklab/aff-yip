import numpy as np

from parse_utils import read_pdb, read_mol2 


class Atom:
    def __init__(self, name, element, coordinates, charge=None, b_factor=None, element_type=None):
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

    def __repr__(self):
        return f"({self.res_name}.{self.chain_id}.{self.res_num})"


class Protein:
    def __init__(self, pdb_filepath=None, read_water=False, read_ligand=False, excl_aa_types=None, excl_chain=None):
        self.pdb_filepath = pdb_filepath
        self.name = pdb_filepath.split('/')[-1].split('.')[0] if pdb_filepath else None
        self.residues = []

        # Optional parameters for filtering residues
        self.read_water = read_water
        self.read_ligand = read_ligand
        self.excl_aa_types = excl_aa_types if excl_aa_types is not None else []
        self.excl_chain = excl_chain if excl_chain is not None else []

        if pdb_filepath:
            self._load_pdb()

    def _load_pdb(self):
        # ... logic to create Residue and Atom objects and link them ...
        res_names, res_chains, xyz, atms, elems, lig_flag = read_pdb(self.pdb_filepath, 
                                                      read_ligand=self.read_ligand, 
                                                      read_water=self.read_water, 
                                                      excl_aa_types=self.excl_aa_types, 
                                                      excl_chain=self.excl_chain)
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
        for elem, q, xyz, atm, atype in zip(elems, qs, xyzs, atms, atypes):
            atom = Atom(name=atm, element=elem, coordinates=xyz, charge=q, element_type=atype)
            self.atoms.append(atom)

        for bond, border in zip(bonds, borders):
            atom1 = self.atoms[bond[0]]
            atom2 = self.atoms[bond[1]]
            self.bonds.append((atom1, atom2, border))


class VirtualNode:
    def __init__(self, coordinates, node_type='virtual'):
        self.coordinates = np.array(coordinates)
        self.node_type = node_type  # e.g., 'virtual', 'pocket', etc.
    
    def get_water_occupancy(self, water_coords):
        """Calculate water occupancy based on distance to water molecules."""
        distances = np.linalg.norm(self.coordinates - water_coords, axis=1)
        occupancy = np.sum(distances < 3.5)

    def __repr__(self):
        return f"VirtualNode(type={self.node_type}, coords={self.coordinates})"
    

if __name__ == '__main__':
    # Example usage
    pdb_file = '5hz8.pdb'
    mol2_file = '/home/j2ho/DB/pdbbind/v2020-refined/5hz8/5hz8_ligand.mol2'
    protein = Protein(pdb_filepath=pdb_file)
    ligand = Ligand(mol2_filepath=mol2_file)