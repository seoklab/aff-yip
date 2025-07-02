import numpy as np
import src.data.featurizers.const as const 
from pathlib import Path

def read_pdb(pdb:Path, read_ligand=False, read_water=False,
             excl_aa_types=[], excl_chain=[], read_chain=[]):
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
        atm = l[12:16].strip()
        aa3 = l[17:20].strip() # doesn't handle duplicates
        # ATOM     99  CB ALYS A  10       1.062  -0.800  -7.661  0.70 12.80           C  
        # ATOM    100  CB BLYS A  10       1.120  -0.790  -7.639  0.30 13.00           C  
        # adds two CB atoms ... 
        if len(l) >= 76:
            elem = l[76:].strip()
        else: 
            elem = None 
        if not read_ligand and read_water:
            if l.startswith('HETATM'): 
                if not aa3 in ['HOH', 'DOD']: 
                    continue

        if excl_aa_types != [] and aa3 in excl_aa_types: continue
        if excl_chain: 
            if l[21] in excl_chain: continue
        if read_chain !=[] and l[21] not in read_chain: continue
        reschain = l[21]+'.'+l[22:27].strip() # resnum sometimes can be negative. e.g.) 'A.-1' 
        # (e.g.) atm: OD1 / aa3: ASP / reschain A.72

        if aa3[:2] in const.METAL: aa3 = aa3[:2]
        if aa3 in const.AMINOACID:
            if atm == 'CA':
                if reschain not in reschains:
                    resnames.append(aa3)
                    reschains.append(reschain)
        elif aa3 in const.NUCLEICACID:
            if atm == "C1'":
                if reschain not in reschains:
                    resnames.append(aa3)
                    reschains.append(reschain)
        elif aa3 in const.METAL:
            if reschain not in reschains:
                resnames.append(aa3)
                reschains.append(reschain)
        elif read_ligand and reschain not in reschains:
            if reschain not in reschains:
                resnames.append(aa3)
                reschains.append(reschain)
        elif read_water and reschain not in reschains: 
            if reschain not in reschains:
                resnames.append(aa3)
                reschains.append(reschain)

        if reschain not in xyz:
            xyz[reschain] = {}
            atms[reschain] = []
            elems[reschain] = {}
        if atm not in xyz[reschain].keys(): 
            xyz[reschain][atm] = np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
        ## ALYS, BLYS 다 중복으로 더했던 방식 
        # atms[reschain].append(atm)
        #elems[reschain][atm] = elem if elem is not None else 'X'  # Default to 'X' if no element info
        # for now 이렇게 해서 중복 방지 
        if atm not in atms[reschain]:
            atms[reschain].append(atm)
            elems[reschain][atm] = elem if elem is not None else 'X'  # Default to 'X' if no element info
        lig_flag[reschain] = is_lig
    return resnames, reschains, xyz, atms, elems, lig_flag


if __name__ == '__main__':
    # Example usage
    pdb_file = '/home/j2ho/DB/pdbbind/v2020-refined/5hz8/5hz8_protein.pdb'
    pdb_file = '5hz8.pdb'
    rec_resnames, rec_reschains, rec_xyzs, rec_atms, rec_elems, lig_flag = read_pdb(pdb_file, read_ligand=True, read_water=True)
    print("Protein Residues:", rec_resnames)
    print("Protein Chains:", rec_reschains)
    print("Protein Coordinates:", rec_xyzs)
    print("Protein Atoms:", rec_atms)
    print ("Protein Elements:", rec_elems)