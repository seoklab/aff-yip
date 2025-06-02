from src.data.structures import VirtualNode

def write_virtual_nodes_pdb(virtual_nodes:list[VirtualNode], filepath="virtual_nodes.pdb", element="C", chain_id="X"):
    """
    Write virtual nodes to a PDB file as dummy atoms for visualization.
    
    Args:
        virtual_nodes (List[VirtualNode]): list of VirtualNode objects
        filepath (str): output .pdb file path
        element (str): dummy atom type (e.g., 'C', 'H', 'X')
        chain_id (str): chain ID to group the virtual nodes
    """
    with open(filepath, 'w') as f:
        for i, node in enumerate(virtual_nodes, 1):
            x, y, z = node.coordinates
            f.write(
                "HETATM{atom_id:5d}  {element:^2}  VRT {chain_id}   1    "
                "{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2}\n".format(
                    atom_id=i, x=x, y=y, z=z, element=element, chain_id=chain_id
                )
            )
        f.write("END\n")
    return filepath