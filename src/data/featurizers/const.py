
ELEMS = ['X','H','C','N','O','Cl','F','I','Br','P','S'] #0 index goes to "empty node"

# Tip atom definitions
AA_to_tip = {"ALA":"CB", "CYS":"SG", "ASP":"CG", "ASN":"CG", "GLU":"CD",
             "GLN":"CD", "PHE":"CZ", "HIS":"NE2", "ILE":"CD1", "GLY":"CA",
             "LEU":"CG", "MET":"SD", "ARG":"CZ", "LYS":"NZ", "PRO":"CG",
             "VAL":"CB", "TYR":"OH", "TRP":"CH2", "SER":"OG", "THR":"OG1"}

# Residue number definition
AMINOACID = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',\
             'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',\
             'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
# Residue number to index mapping
residuemap = dict([(AMINOACID[i], i) for i in range(len(AMINOACID))])
NUCLEICACID = ['ADE','CYT','GUA','THY','URA'] #nucleic acids

METAL = ['CA','ZN','MN','MG','FE','CD','CO','CU']
ALL_AAS = ['UNK'] + AMINOACID + NUCLEICACID + METAL
NMETALS = len(METAL)

N_AATYPE = len(ALL_AAS)

# minimal sc atom representation (Nx8)
aa2short={
    "ALA": (" N  "," CA "," C  "," CB ",  None,  None,  None,  None), 
    "ARG": (" N  "," CA "," C  "," CB "," CG "," CD "," NE "," CZ "), 
    "ASN": (" N  "," CA "," C  "," CB "," CG "," OD1",  None,  None), 
    "ASP": (" N  "," CA "," C  "," CB "," CG "," OD1",  None,  None), 
    "CYS": (" N  "," CA "," C  "," CB "," SG ",  None,  None,  None), 
    "GLN": (" N  "," CA "," C  "," CB "," CG "," CD "," OE1",  None), 
    "GLU": (" N  "," CA "," C  "," CB "," CG "," CD "," OE1",  None), 
    "GLY": (" N  "," CA "," C  ",  None,  None,  None,  None,  None), 
    "HIS": (" N  "," CA "," C  "," CB "," CG "," ND1",  None,  None),
    "ILE": (" N  "," CA "," C  "," CB "," CG1"," CD1",  None,  None), 
    "LEU": (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None), 
    "LYS": (" N  "," CA "," C  "," CB "," CG "," CD "," CE "," NZ "), 
    "MET": (" N  "," CA "," C  "," CB "," CG "," SD "," CE ",  None), 
    "PHE": (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None),
    "PRO": (" N  "," CA "," C  "," CB "," CG "," CD ",  None,  None), 
    "SER": (" N  "," CA "," C  "," CB "," OG ",  None,  None,  None),
    "THR": (" N  "," CA "," C  "," CB "," OG1",  None,  None,  None),
    "TRP": (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None),
    "TYR": (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None),
    "VAL": (" N  "," CA "," C  "," CB "," CG1",  None,  None,  None),
}

# Atom types:
atypes = {('ALA', 'CA'): 'CAbb', ('ALA', 'CB'): 'CH3', ('ALA', 'C'): 'CObb', ('ALA', 'N'): 'Nbb', ('ALA', 'O'): 'OCbb', ('ARG', 'CA'): 'CAbb', ('ARG', 'CB'): 'CH2', ('ARG', 'C'): 'CObb', ('ARG', 'CD'): 'CH2', ('ARG', 'CG'): 'CH2', ('ARG', 'CZ'): 'aroC', ('ARG', 'NE'): 'Narg', ('ARG', 'NH1'): 'Narg', ('ARG', 'NH2'): 'Narg', ('ARG', 'N'): 'Nbb', ('ARG', 'O'): 'OCbb', ('ASN', 'CA'): 'CAbb', ('ASN', 'CB'): 'CH2', ('ASN', 'C'): 'CObb', ('ASN', 'CG'): 'CNH2', ('ASN', 'ND2'): 'NH2O', ('ASN', 'N'): 'Nbb', ('ASN', 'OD1'): 'ONH2', ('ASN', 'O'): 'OCbb', ('ASP', 'CA'): 'CAbb', ('ASP', 'CB'): 'CH2', ('ASP', 'C'): 'CObb', ('ASP', 'CG'): 'COO', ('ASP', 'N'): 'Nbb', ('ASP', 'OD1'): 'OOC', ('ASP', 'OD2'): 'OOC', ('ASP', 'O'): 'OCbb', ('CYS', 'CA'): 'CAbb', ('CYS', 'CB'): 'CH2', ('CYS', 'C'): 'CObb', ('CYS', 'N'): 'Nbb', ('CYS', 'O'): 'OCbb', ('CYS', 'SG'): 'S', ('GLN', 'CA'): 'CAbb', ('GLN', 'CB'): 'CH2', ('GLN', 'C'): 'CObb', ('GLN', 'CD'): 'CNH2', ('GLN', 'CG'): 'CH2', ('GLN', 'NE2'): 'NH2O', ('GLN', 'N'): 'Nbb', ('GLN', 'OE1'): 'ONH2', ('GLN', 'O'): 'OCbb', ('GLU', 'CA'): 'CAbb', ('GLU', 'CB'): 'CH2', ('GLU', 'C'): 'CObb', ('GLU', 'CD'): 'COO', ('GLU', 'CG'): 'CH2', ('GLU', 'N'): 'Nbb', ('GLU', 'OE1'): 'OOC', ('GLU', 'OE2'): 'OOC', ('GLU', 'O'): 'OCbb', ('GLY', 'CA'): 'CAbb', ('GLY', 'C'): 'CObb', ('GLY', 'N'): 'Nbb', ('GLY', 'O'): 'OCbb', ('HIS', 'CA'): 'CAbb', ('HIS', 'CB'): 'CH2', ('HIS', 'C'): 'CObb', ('HIS', 'CD2'): 'aroC', ('HIS', 'CE1'): 'aroC', ('HIS', 'CG'): 'aroC', ('HIS', 'ND1'): 'Nhis', ('HIS', 'NE2'): 'Ntrp', ('HIS', 'N'): 'Nbb', ('HIS', 'O'): 'OCbb', ('ILE', 'CA'): 'CAbb', ('ILE', 'CB'): 'CH1', ('ILE', 'C'): 'CObb', ('ILE', 'CD1'): 'CH3', ('ILE', 'CG1'): 'CH2', ('ILE', 'CG2'): 'CH3', ('ILE', 'N'): 'Nbb', ('ILE', 'O'): 'OCbb', ('LEU', 'CA'): 'CAbb', ('LEU', 'CB'): 'CH2', ('LEU', 'C'): 'CObb', ('LEU', 'CD1'): 'CH3', ('LEU', 'CD2'): 'CH3', ('LEU', 'CG'): 'CH1', ('LEU', 'N'): 'Nbb', ('LEU', 'O'): 'OCbb', ('LYS', 'CA'): 'CAbb', ('LYS', 'CB'): 'CH2', ('LYS', 'C'): 'CObb', ('LYS', 'CD'): 'CH2', ('LYS', 'CE'): 'CH2', ('LYS', 'CG'): 'CH2', ('LYS', 'N'): 'Nbb', ('LYS', 'NZ'): 'Nlys', ('LYS', 'O'): 'OCbb', ('MET', 'CA'): 'CAbb', ('MET', 'CB'): 'CH2', ('MET', 'C'): 'CObb', ('MET', 'CE'): 'CH3', ('MET', 'CG'): 'CH2', ('MET', 'N'): 'Nbb', ('MET', 'O'): 'OCbb', ('MET', 'SD'): 'S', ('PHE', 'CA'): 'CAbb', ('PHE', 'CB'): 'CH2', ('PHE', 'C'): 'CObb', ('PHE', 'CD1'): 'aroC', ('PHE', 'CD2'): 'aroC', ('PHE', 'CE1'): 'aroC', ('PHE', 'CE2'): 'aroC', ('PHE', 'CG'): 'aroC', ('PHE', 'CZ'): 'aroC', ('PHE', 'N'): 'Nbb', ('PHE', 'O'): 'OCbb', ('PRO', 'CA'): 'CAbb', ('PRO', 'CB'): 'CH2', ('PRO', 'C'): 'CObb', ('PRO', 'CD'): 'CH2', ('PRO', 'CG'): 'CH2', ('PRO', 'N'): 'Npro', ('PRO', 'O'): 'OCbb', ('SER', 'CA'): 'CAbb', ('SER', 'CB'): 'CH2', ('SER', 'C'): 'CObb', ('SER', 'N'): 'Nbb', ('SER', 'OG'): 'OH', ('SER', 'O'): 'OCbb', ('THR', 'CA'): 'CAbb', ('THR', 'CB'): 'CH1', ('THR', 'C'): 'CObb', ('THR', 'CG2'): 'CH3', ('THR', 'N'): 'Nbb', ('THR', 'OG1'): 'OH', ('THR', 'O'): 'OCbb', ('TRP', 'CA'): 'CAbb', ('TRP', 'CB'): 'CH2', ('TRP', 'C'): 'CObb', ('TRP', 'CD1'): 'aroC', ('TRP', 'CD2'): 'aroC', ('TRP', 'CE2'): 'aroC', ('TRP', 'CE3'): 'aroC', ('TRP', 'CG'): 'aroC', ('TRP', 'CH2'): 'aroC', ('TRP', 'CZ2'): 'aroC', ('TRP', 'CZ3'): 'aroC', ('TRP', 'NE1'): 'Ntrp', ('TRP', 'N'): 'Nbb', ('TRP', 'O'): 'OCbb', ('TYR', 'CA'): 'CAbb', ('TYR', 'CB'): 'CH2', ('TYR', 'C'): 'CObb', ('TYR', 'CD1'): 'aroC', ('TYR', 'CD2'): 'aroC', ('TYR', 'CE1'): 'aroC', ('TYR', 'CE2'): 'aroC', ('TYR', 'CG'): 'aroC', ('TYR', 'CZ'): 'aroC', ('TYR', 'N'): 'Nbb', ('TYR', 'OH'): 'OH', ('TYR', 'O'): 'OCbb', ('VAL', 'CA'): 'CAbb', ('VAL', 'CB'): 'CH1', ('VAL', 'C'): 'CObb', ('VAL', 'CG1'): 'CH3', ('VAL', 'CG2'): 'CH3', ('VAL', 'N'): 'Nbb', ('VAL', 'O'): 'OCbb'}

# Atome type to index
atype2num = {'CNH2': 0, 'Npro': 1, 'CH1': 2, 'CH3': 3, 'CObb': 4, 'aroC': 5, 'OOC': 6, 'Nhis': 7, 'Nlys': 8, 'COO': 9, 'NH2O': 10, 'S': 11, 'Narg': 12, 'OCbb': 13, 'Ntrp': 14, 'Nbb': 15, 'CH2': 16, 'CAbb': 17, 'ONH2': 18, 'OH': 19}

gentype2num = {'CS':0, 'CS1':1, 'CS2':2,'CS3':3,
               'CD':4, 'CD1':5, 'CD2':6,'CR':7, 'CT':8,
               'CSp':9,'CDp':10,'CRp':11,'CTp':12,'CST':13,'CSQ':14,
               'HO':15,'HN':16,'HS':17,
               # Nitrogen
               'Nam':18, 'Nam2':19, 'Nad':20, 'Nad3':21, 'Nin':22, 'Nim':23,
               'Ngu1':24, 'Ngu2':25, 'NG3':26, 'NG2':27, 'NG21':28,'NG22':29, 'NG1':30, 
               'Ohx':31, 'Oet':32, 'Oal':33, 'Oad':34, 'Oat':35, 'Ofu':36, 'Ont':37, 'OG2':38, 'OG3':39, 'OG31':40,
               #S/P
               'Sth':41, 'Ssl':42, 'SR':43,  'SG2':44, 'SG3':45, 'SG5':46, 'PG3':47, 'PG5':48, 
               # Halogens
               'Br':49, 'I':50, 'F':51, 'Cl':52, 'BrR':53, 'IR':54, 'FR':55, 'ClR':56,
               # Metals
               'Ca2p':57, 'Mg2p':58, 'Mn':59, 'Fe2p':60, 'Fe3p':60, 'Zn2p':61, 'Co2p':62, 'Cu2p':63, 'Cd':64}

atomic_radii = {"C":  2.0,"N": 1.5,"O": 1.4,"S": 1.85,"H": 0.0, #ignore hydrogen for consistency
                "F": 1.47,"Cl":1.75,"Br":1.85,"I": 2.0,'P': 1.8}
 