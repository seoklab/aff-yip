def featurize_protein_graph_with_virtual_nodes(self, protein: Protein, center=None, crop_size=None) -> torch_geometric.data.Data:
    # Featurize the protein graph with virtual nodes
    # === Full coordinates ===
    X_res_all = stack_residue_coordinates(protein)
    X_res = X_res_all[1::3]  # CA only
    X_water = stack_water_coordinates(protein)
    has_water = X_water is not None and X_water.numel() > 0

    X_ligand = torch.tensor(self.ligand.get_coordinates())
    virtual_nodes = generate_virtual_nodes(
        self.protein.get_coordinates(), X_ligand.numpy()
    )
    X_virtual = torch.tensor([v.coordinates for v in virtual_nodes], dtype=torch.float32)

    if has_water:
        X_all = torch.cat([X_res, X_water, X_virtual], dim=0)
    else:
        X_all = torch.cat([X_res, X_virtual], dim=0)

    print(f"Protein featurization: {X_res.size(0)} residues, {X_water.size(0) if has_water else 0} water molecules, {X_virtual.size(0)} virtual nodes")

    # === Optional cropping ===
    if center is not None and crop_size is not None:
        mask_res = ((X_res - center).abs() < crop_size / 2).all(dim=-1)
        X_res = X_res[mask_res]
        keep_res_idx = mask_res.nonzero(as_tuple=False).squeeze(-1)

        if has_water:
            mask_water = ((X_water - center).abs() < crop_size / 2).all(dim=-1)
            X_water = X_water[mask_water]
            keep_water_idx = mask_water.nonzero(as_tuple=False).squeeze(-1)
        else:
            keep_water_idx = torch.empty((0,), dtype=torch.long)

        mask_virtual = ((X_virtual - center).abs() < crop_size / 2).all(dim=-1)
        X_virtual = X_virtual[mask_virtual]

        if has_water:
            X_all = torch.cat([X_res, X_water, X_virtual], dim=0)
        else:
            X_all = torch.cat([X_res, X_virtual], dim=0)
    else:
        keep_res_idx = torch.arange(X_res.size(0))
        keep_water_idx = torch.arange(X_water.size(0)) if has_water else torch.empty((0,), dtype=torch.long)

    print(f"After cropping: {X_res.size(0)} residues, {X_water.size(0) if has_water else 0} water molecules, {X_virtual.size(0)} virtual nodes")

    # === Node type ===
    if has_water:
        node_type = torch.cat([
            torch.zeros(X_res.size(0)),
            torch.ones(X_water.size(0)),
            2 * torch.ones(X_virtual.size(0))
        ]).long()
    else:
        node_type = torch.cat([
            torch.zeros(X_res.size(0)),
            2 * torch.ones(X_virtual.size(0))
        ]).long()

    # === Node scalar features ===
    node_s_all_res = get_residue_dihedrals(X_res_all)
    node_s_res = node_s_all_res[keep_res_idx]

    if has_water:
        node_s_water = get_water_embeddings(X_water, num_embeddings=node_s_res.size(1))
        node_s = torch.cat([node_s_res, node_s_water, torch.zeros(X_virtual.size(0), node_s_res.size(1))], dim=0)
    else:
        node_s = torch.cat([node_s_res, torch.zeros(X_virtual.size(0), node_s_res.size(1))], dim=0)

    print(f"Node scalar features: {node_s.size(0)} nodes")

    # === Node vector features ===
    sidechain = get_sidechain_orientation(X_res_all).unsqueeze(-2)
    backbone = get_backbone_orientation(X_res_all)
    node_v_res_all = torch.cat([sidechain, backbone], dim=-2)
    node_v_res = node_v_res_all[keep_res_idx]

    if has_water:
        node_v_water = torch.zeros(X_water.size(0), 3, 3)
        node_v = torch.cat([node_v_res, node_v_water, torch.zeros(X_virtual.size(0), 3, 3)], dim=0)
    else:
        node_v = torch.cat([node_v_res, torch.zeros(X_virtual.size(0), 3, 3)], dim=0)

    # === Edge index and edge features ===
    edge_index = torch_cluster.knn_graph(X_all, k=self.top_k)
    E_vectors = X_all[edge_index[0]] - X_all[edge_index[1]]
    edge_v = _normalize_torch(E_vectors)
    edge_dist = E_vectors.norm(dim=-1)
    edge_s = torch.cat([
        get_rbf(edge_dist, D_count=16, device=edge_index.device),
        get_positional_embeddings(edge_index, num_embeddings=16)
    ], dim=-1)

    # === Edge types ===
    src_type = node_type[edge_index[0]]
    dst_type = node_type[edge_index[1]]
    edge_type_id = src_type * 3 + dst_type
    edge_type_onehot = torch.nn.functional.one_hot(edge_type_id, num_classes=9).float()
    edge_s = torch.cat([edge_s, edge_type_onehot], dim=-1)

    data = torch_geometric.data.Data(
        x=X_all, node_type=node_type,
        node_s=torch.nan_to_num(node_s),
        node_v=torch.nan_to_num(node_v),
        edge_s=torch.nan_to_num(edge_s),
        edge_v=torch.nan_to_num(edge_v),
        edge_index=edge_index
    )

    return data