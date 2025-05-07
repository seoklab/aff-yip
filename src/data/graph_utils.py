# generating ligand and receptor graph
import os
import sys
import random
import torch

from pathlib import Path
import numpy as np
import dgl 
import prody
prody.confProDy(verbosity='none')
from prody import parsePDB


def lig_graph_gen(mol2file):
    
    return lig

def rec_graph_gen(pdbfile, cntr): 
    """ 
    get pocket center and generate pocket graph
    residue + virtual nodes 
    """ 
    return rec