# in src/model/model_threebody_virtual.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.bin

from torch_geometric.nn.pool import graclus
from torch_scatter import scatter_mean

from src.model.gvp_encoder import GVPGraphEncoderHybrid as GVPGraphEncoder
from src.model.my_modules import MLPCoordinatePredictor_SidechainMap as MLPStructModule
from src.model.structure_modules.egnn_module import EGNNCoordinatePredictor_SidechainMap as EGNNStructModule

from src.model.loss_utils import AffHuberLoss as AffinityLoss
from src.model.loss_utils import SidechainMapCoordinateLoss as StrLoss

class AFFModel_ThreeBody(pl.LightningModule):
    def __init__(self,
                 protein_node_dims=(26, 3), # 6 dihedral angles + 20 amino acid types & 3 vectors
                 virtual_node_dims=(26, 3), # 1 water occupancy + padding & 3 vectors (now zero vectors)
                 protein_edge_dims=(41, 1), # 41 edge scalar & 1 edge vector (simply X1-X2 norm vector)
                 ligand_node_dims=(46, 0), # 46 ligand atom types & 0 vectors
                 ligand_edge_dims=(9, 0), # 9 edge scalar & 0 edge vector
                 protein_hidden_dims=(196, 16),
                 virtual_hidden_dims=(196, 3),
                 ligand_hidden_dims=(196, 3),
                 num_gvp_layers=3,
                 dropout=0.1,
                 lr=1e-3,
                 interaction_mode="hierarchical",
                 predict_str=False,
                 str_model_type="egnn", #"mlp",  # "mlp" or "egnn"
                 str_loss_weight=0.3,
                 str_loss_params=None,
                 loss_type="single",  # single: no classification head
                 loss_params=None):
        super().__init__()
        self.save_hyperparameters()

        # === Original model components ===
        self.interaction_mode = interaction_mode
        self.predict_str = predict_str
        self.str_model_type = str_model_type
        # Encoders

        self.protein_encoder = GVPGraphEncoder(
            node_dims=protein_node_dims,
            edge_dims=protein_edge_dims,
            hidden_dims=(64,16),
            num_layers=2,
        )

        self.virtual_encoder = GVPGraphEncoder(
            node_dims=virtual_node_dims,
            edge_dims=protein_edge_dims,  # Use same edge dims as protein
            hidden_dims=(64,16),
            num_layers=2,
        )

        self.protein_virtual_encoder = GVPGraphEncoder(
            node_dims=(64,16), #protein_node_dims,
            edge_dims=protein_edge_dims,
            hidden_dims=protein_hidden_dims,
            num_layers=num_gvp_layers,
            drop_rate=dropout
        )

        self.ligand_encoder = GVPGraphEncoder(
            node_dims=ligand_node_dims,
            edge_dims=ligand_edge_dims,
            hidden_dims=ligand_hidden_dims,
            num_layers=num_gvp_layers,
            drop_rate=dropout
        )

        # Three-Body Interaction Modules
        embed_dim = protein_hidden_dims[0]

        if interaction_mode == "hierarchical":
            self.protein_vn_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=4, batch_first=True
            )
            self.complex_ligand_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=4, batch_first=True
            )
            self.pv_fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim)
            )
            self.plv_fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim)
            )

        # Structure Prediction Setups
        if predict_str:
            self.str_loss_weight = str_loss_weight
            if self.str_model_type == "egnn":
                self.structure_predictor = EGNNStructModule(
                    lig_embed_dim=ligand_hidden_dims[0],
                    prot_embed_dim=embed_dim,
                    hidden_dim=embed_dim,
                    num_layers=3,
                    dropout=dropout
                )
            if self.str_model_type == "mlp":
                self.structure_predictor = MLPStructModule(
                    lig_embed_dim=ligand_hidden_dims[0],
                    prot_embed_dim=embed_dim,
                    hidden_dim=embed_dim,
                )
            # process feature from structure module using MLP
            self.str_feature_mlp = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim)
            )

        # Main regression head
        if predict_str:
            embed_dim *= 2
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        # Additional heads for multitask: classification and regression
        if loss_type == "multitask":
            self.thresholds = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
            class_num = len(self.thresholds) + 1  # +1 for the "extreme" class
            self.classification_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, class_num)
            )

        # Initialize loss function
        self._init_loss_function(loss_type, loss_params)

        # Track  metrics
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

    def _init_loss_function(self, loss_type, loss_params):
        """Initialize the appropriate loss function"""
        if loss_params is None:
            loss_params = {}

        if loss_type == "multitask":
            from src.model.loss_utils import WeightedCategoryLoss_v2
            self.aff_loss_fn = WeightedCategoryLoss_v2(
                    thresholds = self.thresholds,
                    regression_weight=0.75,
                    category_penalty_weight=0.15,
                    extreme_penalty_weight=0,
                    pearson_penalty_weight=0.1,      # Optional
                    relative_error_weight=0.1,       # Optional
                    extreme_boost_low=1.3,
                    extreme_boost_high=1.4
                )
        elif loss_type == "single":
            self.aff_loss_fn = AffinityLoss(
                beta=1.0,
                extreme_weight=1.5,
                ranking_weight=0.5)

        if self.predict_str:
            self.str_loss_fn = StrLoss(
                loss_type="mse",
                ligand_weight=1,
                sidechain_weight=0.3,
                use_distance_loss=True,
                ligand_distance_weight=0.3,
                sidechain_distance_weight=0,
                distance_loss_type="mse",
                distance_cutoff=5.0
            )

        self.loss_type = loss_type

    def forward(self, batch):
        interaction_results = self.get_embeddings(batch)
        complex_embedding = interaction_results['complex_embedding']  # [B, embed_dim]
        results = dict()

        if self.predict_str:
            coord_results, h = self._predict_coordinates_from_shared_embeddings(batch, interaction_results)
            # if coord_results['sidechain_predictions'] is not None:
            results.update(coord_results)
            # Concatenate complex embedding with feature from structure predictor
            complex_embedding = torch.cat([complex_embedding, h], dim=-1)  # [B, embed_dim * 2]

        if self.loss_type == "multitask":
            # Regression prediction
            pred_affinity = self.regressor(complex_embedding).squeeze()
            # Classification prediction
            pred_logits = self.classification_head(complex_embedding)
            # Get true affinities
            affinities = batch['affinity'].to(self.device)
            results_aff = {
                'affinity': pred_affinity,
                'logits': pred_logits,
                'target_affinity': affinities
            }
            results.update(results_aff)
        else:
            # Single-task regression
            pred_affinity = self.regressor(complex_embedding).squeeze()
            affinities = batch['affinity'].to(self.device)
            results_aff = {
                'affinity': pred_affinity,
                'target_affinity': affinities
            }
            results.update(results_aff)


        return results

    def get_embeddings(self, batch):
        """Get embeddings before final prediction (for multi-task learning)"""
        protein_virtual_batch = batch['protein_virtual'].to(self.device)
        ligand_batch = batch['ligand'].to(self.device)

        # Split protein and water nodes
        protein_mask = protein_virtual_batch.node_type == 0
        virtual_mask = protein_virtual_batch.node_type == 1
        # water_mask = protein_batch.node_type == 2

        # Step1: Process prot-virtual /lig component
        prot_s, prot_v, virtual_s, virtual_v = self._process_protein_and_virtual_nodes(protein_virtual_batch, protein_mask, virtual_mask)
        lig_s = self._process_ligand_nodes(ligand_batch)

        # Step2: Pool virtual nodes
        virtual_coords_initial = protein_virtual_batch.x[virtual_mask]
        virtual_batch_idx_initial = protein_virtual_batch.batch[virtual_mask]
        v2v_edge_index = self._get_virtual_subgraph(protein_virtual_batch, virtual_mask)

        virtual_s, virtual_coords, virtual_batch_idx = self._pool_virtual_nodes(
            s=virtual_s,
            coords=virtual_coords_initial,
            edge_index=v2v_edge_index,
            batch_idx=virtual_batch_idx_initial)
        # Step3: Get interaction embeddings
        complex_embeddings, ligand_embeddings_after_interaction, virtual_embeddings_after_interaction, protein_embeddings_after_interaction = self._get_hierarchical_interaction_embeddings(
            prot_s, virtual_s, lig_s,
            protein_batch_idx=protein_virtual_batch.batch[protein_mask],
            virtual_batch_idx=virtual_batch_idx,
            ligand_batch_idx=ligand_batch.batch
        )

        return {
            'complex_embedding': complex_embeddings,           # [B, embed_dim] for affinity
            'virtual_embeddings_f': virtual_embeddings_after_interaction,  # [N_virtual, embed_dim] for coords
            'virtual_batch_idx': virtual_batch_idx,   # Pooled index
            'virtual_coords': virtual_coords,          # Grid coordinates
            'ligand_embeddings_f': ligand_embeddings_after_interaction,  # [N_ligand, embed_dim]
            'ligand_batch_idx': ligand_batch.batch,  # Batch indices for ligands
            'protein_batch_idx': protein_virtual_batch.batch[protein_mask],  # ← FIX: Only protein nodes
            'protein_embeddings_f': protein_embeddings_after_interaction,  # [N_protein, embed_dim]
            'backbone_coords': protein_virtual_batch.x[protein_mask],
        }

    def _get_hierarchical_interaction_embeddings(self, prot_s, virtual_s, lig_s,
                                                      protein_batch_idx, virtual_batch_idx, ligand_batch_idx):
        """Get_embeddings-Step3: Hierarchical interaction but track virtual embeddings for coordinate prediction"""

        # Get batch information
        prot_batch_sizes = torch.bincount(protein_batch_idx).tolist() if protein_batch_idx.numel() > 0 else []
        virtual_batch_sizes = torch.bincount(virtual_batch_idx).tolist() if virtual_batch_idx.numel() > 0 else []
        lig_batch_sizes = torch.bincount(ligand_batch_idx).tolist()

        # Split by batch
        prot_s_list = torch.split(prot_s, prot_batch_sizes) if prot_s.size(0) > 0 else []
        virtual_s_list = torch.split(virtual_s, virtual_batch_sizes) if virtual_s.size(0) > 0 else []
        lig_s_list = torch.split(lig_s, lig_batch_sizes)

        complex_embeddings = []
        ligand_embeddings_after_interaction = []
        virtual_embeddings_after_interaction = []
        protein_embeddings_after_interaction = []
        batch_size = len(lig_s_list)

        for i in range(batch_size):
            l = lig_s_list[i]
            p = prot_s_list[i] if i < len(prot_s_list) else torch.empty(0, self.hparams.protein_hidden_dims[0], device=self.device)
            v = virtual_s_list[i] if i < len(virtual_s_list) else torch.empty(0, self.hparams.protein_hidden_dims[0], device=self.device)
            # Hierarchical interaction with virtual tracking
            complex_emb, ligand_after_interaction, virtual_after_interaction, protein_after_interaction = self._hierarchical_interaction_with_virtual(p, v, l)
            complex_embeddings.append(complex_emb) # [embed_dim]
            ligand_embeddings_after_interaction.append(ligand_after_interaction) # [N_ligand, embed_dim]
            virtual_embeddings_after_interaction.append(virtual_after_interaction) # [N_virtual_pooled, embed_dim]
            protein_embeddings_after_interaction.append(protein_after_interaction) # [N_protein, embed_dim]

        # Stack results
        complex_embeddings = torch.stack(complex_embeddings)  # [B, embed_dim]

        # Concatenate virtual embeddings (maintaining batch order)
        if virtual_embeddings_after_interaction and all(v.size(0) > 0 for v in virtual_embeddings_after_interaction):
            virtual_embeddings_after_interaction = torch.cat(virtual_embeddings_after_interaction, dim=0)
        else:
            virtual_embeddings_after_interaction = virtual_s  # Fallback to input

        # Concatenate protein embeddings (maintaining batch order)
        if protein_embeddings_after_interaction and all(p.size(0) > 0 for p in protein_embeddings_after_interaction):
            protein_embeddings_after_interaction = torch.cat(protein_embeddings_after_interaction, dim=0)
        else:
            protein_embeddings_after_interaction = prot_s

        # Concatenate ligand embeddings (maintaining batch order)
        if ligand_embeddings_after_interaction and all(l.size(0) > 0 for l in ligand_embeddings_after_interaction):
            ligand_embeddings_after_interaction = torch.cat(ligand_embeddings_after_interaction, dim=0)
        else:
            ligand_embeddings_after_interaction = lig_s

        return complex_embeddings, ligand_embeddings_after_interaction, virtual_embeddings_after_interaction, protein_embeddings_after_interaction

    def _hierarchical_interaction_with_virtual(self, p, v, l):
        """Hierarchical: (P+V) -> complex -> interact with L"""
        embed_dim = self.hparams.protein_hidden_dims[0]

        # Step 1: Protein-VirtualNode interaction (if both exist)
        if p.size(0) > 0 and v.size(0) > 0:
            # Cross-attention between protein and virtual nodes
            p_batch = p.unsqueeze(0)  # [1, N_p, D]
            v_batch = v.unsqueeze(0)  # [1, N_v, D]

            # virtual node attends to protein, protein attends to virtual node
            v_enhanced, _ = self.protein_vn_attn(v_batch, p_batch, p_batch)  # [1, N_v, D]
            p_enhanced, _ = self.protein_vn_attn(p_batch, v_batch, v_batch)  # [1, N_p, D]

            # Pool and fuse for complex representation
            p_pooled = p_enhanced.mean(dim=1)  # [1, D]
            v_pooled = v_enhanced.mean(dim=1)  # [1, D]

            pv_combined = torch.cat([p_pooled, v_pooled], dim=-1)  # [1, 2*D]
            complex_repr = self.pv_fusion(pv_combined).unsqueeze(1)  # [1, 1, D]

            virtual_after_interaction = v_enhanced.squeeze(0)  # [N_v, D]
            protein_after_interaction = p_enhanced.squeeze(0)  # [N_p, D]

        elif p.size(0) > 0:
            # print ("Warning: Only protein nodes present, no virtual nodes.")
            # Only protein - need to add batch dimension properly
            complex_repr = p.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, D]
            protein_after_interaction = p
            virtual_after_interaction = torch.empty(0, embed_dim, device=self.device)
        elif v.size(0) > 0:
            # Only water - need to add batch dimension properly
            complex_repr = v.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, D]
            virtual_after_interaction = v  # Use virtual node embeddings given as input
            protein_after_interaction = torch.empty(0, embed_dim, device=self.device)
        else:
            # Neither protein nor water - create properly shaped zero tensor
            complex_repr = torch.zeros(1, 1, embed_dim, device=self.device)  # [1, 1, D]
            virtual_after_interaction = torch.empty(0, embed_dim, device=self.device)
            protein_after_interaction = torch.empty(0, embed_dim, device=self.device)

        # Step 2: Complex-Ligand interaction
        if l.size(0) > 0:
            l_batch = l.unsqueeze(0)  # [1, N_l, D]
            # Ligand attends to protein-water complex
            l_enhanced, _ = self.complex_ligand_attn(l_batch, complex_repr, complex_repr)  # [1, N_l, D]
            ligand_after_interaction = l_enhanced.squeeze(0)  # [N_l, D]
            l_pooled = l_enhanced.mean(dim=1)  # [1, D]
            # concat all embeddings and linear
            plv_combined = torch.cat([complex_repr.squeeze(0), l_pooled], dim=-1) # [1, 2*D]
            final_repr = self.plv_fusion(plv_combined)  # [1, 1, D]
        else:
            final_repr = complex_repr.squeeze(1)  # Remove sequence dimension: [1, D]

        return final_repr.squeeze(0), ligand_after_interaction, virtual_after_interaction, protein_after_interaction # [D]

    def _predict_coordinates_from_shared_embeddings(self, batch, interaction_results):
        """Predict ligand and sidechain coordinates using final embeddings"""
        ligand_batch=batch['ligand']
        ligand_embeddings = interaction_results['ligand_embeddings_f']
        ligand_batch_idx = interaction_results['ligand_batch_idx']
        protein_embeddings = interaction_results['protein_embeddings_f']  # [N_protein, embed_dim]
        protein_batch_idx = interaction_results['protein_batch_idx']  # [N_protein]
        protein_virtual_batch = batch['protein_virtual']
        sidechain_map = batch['sidechain_map']

        result, h_dict = self.structure_predictor(
            ligand_embeddings=ligand_embeddings,
            ligand_batch_idx=ligand_batch_idx,
            protein_embeddings=protein_embeddings,
            protein_batch_idx=protein_batch_idx,
            protein_virtual_batch=protein_virtual_batch,
            ligand_batch=ligand_batch,
            sidechain_map=sidechain_map
        )
        h = []
        for batch_id in torch.unique(ligand_batch_idx):
            lig_h = h_dict['ligand_features'].get(batch_id.item(), None)
            protein_h = h_dict['protein_features'].get(batch_id.item(), None)
            sidechain_h = h_dict['sidechain_features'].get(batch_id.item(), None)
            lig_h = lig_h.mean(dim=0, keepdim=True)
            protein_h = protein_h.mean(dim=0, keepdim=True)
            sidechain_h = sidechain_h.mean(dim=0, keepdim=True)
            h_idx = torch.cat([lig_h, protein_h, sidechain_h], dim=-1)  # [1, emb_dim * 3]
            h_idx = self.str_feature_mlp(h_idx)  # [1, emb_dim]
            h.append(h_idx)
        # stack h
        h = torch.cat(h, dim=0)  # [N_differentforitem, emb_dim * 3]
        return result, h

    # ========================================
    # LOSS COMPUTATION AND TRAINING STEPS
    # ========================================

    def _compute_loss(self, results):
        """Separate affinity and coordinate losses - much cleaner!"""
        loss_dict = {}

        # ===== AFFINITY LOSS (unchanged) =====
        if self.loss_type == "multitask":
            affinity_result = self.aff_loss_fn(
                results['affinity'],
                results['target_affinity'],
                results['logits']
            )
            if isinstance(affinity_result, tuple):
                total_affinity_loss, reg_loss, cat_penalty, extreme_penalty, pearson_pen = affinity_result
                loss_dict.update({
                    'affinity_loss': total_affinity_loss,
                    'reg_loss': reg_loss,
                    'category_loss': cat_penalty,
                    'extreme_penalty': extreme_penalty,
                    'ranking_loss': pearson_pen
                })
        else:
            affinity_result = self.aff_loss_fn(results['affinity'], results['target_affinity'])
            if isinstance(affinity_result, tuple):
                # reg_loss = weighted huber
                # ranking loss = 1 - correlation
                total_affinity_loss, reg_loss, ranking_loss, _, _ = affinity_result
                loss_dict.update({
                    'affinity_loss': total_affinity_loss,
                    'reg_loss': reg_loss,
                    'ranking_loss': ranking_loss
                })

        total_loss = total_affinity_loss

        # ===== COORDINATE LOSS =====
        if self.predict_str and 'sidechain_predictions' in results:
            # Use the new SidechainMapCoordinateLoss
            str_loss, str_loss_dict = self.str_loss_fn(results)

            weighted_str_loss = self.str_loss_weight * str_loss
            total_loss = total_loss +  weighted_str_loss

            # Add coordinate losses to loss dict
            loss_dict.update(str_loss_dict)
            loss_dict['weighted_str_loss'] = weighted_str_loss.item()
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def training_step(self, batch, batch_idx):
        results = self(batch)
        total_loss, loss_dict = self._compute_loss(results)
        # Log loss components
        actual_batch_size = len(batch['ligand'])
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':
                self.log(f"train_{loss_name}", loss_value, batch_size=actual_batch_size)

        # Store predictions for epoch-level metrics
        pred_affinities = results['affinity'].detach().cpu().flatten()
        target_affinities = results['target_affinity'].detach().cpu().flatten()
        self.train_predictions.extend(pred_affinities.tolist())
        self.train_targets.extend(target_affinities.tolist())

        # self.train_predictions.extend(results['affinity'].detach().cpu().tolist())
        # self.train_targets.extend(results['target_affinity'].detach().cpu().tolist())

        # Log predictions for debugging
        self._log_predictions(batch, results['affinity'], results['target_affinity'], "Train")

        # Log main loss
        self.log("train_loss", total_loss, prog_bar=True, batch_size=actual_batch_size)

        return total_loss

    def validation_step(self, batch, batch_idx):
        results = self(batch)
        total_loss, loss_dict = self._compute_loss(results)

        # Log loss components
        actual_batch_size = len(batch['ligand'])
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':
                self.log(f"val_{loss_name}", loss_value, batch_size=actual_batch_size,
                        on_epoch=True, sync_dist=True)

        # Store predictions for epoch-level metrics
        pred_affinities = results['affinity'].detach().cpu().flatten()
        target_affinities = results['target_affinity'].detach().cpu().flatten()
        self.val_predictions.extend(pred_affinities.tolist())
        self.val_targets.extend(target_affinities.tolist())
        # self.val_predictions.extend(results['affinity'].detach().cpu().tolist())
        # self.val_targets.extend(results['target_affinity'].detach().cpu().tolist())

        # Compute additional metrics
        pred_affinity = results['affinity']
        affinities = results['target_affinity']
        mae = torch.mean(torch.abs(pred_affinity - affinities))

        # Log metrics
        self.log("val_loss", total_loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=actual_batch_size)
        self.log("val_mae", mae, prog_bar=True, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)

        # Log predictions for debugging
        self._log_predictions(batch, results['affinity'], results['target_affinity'], "Valid")

        return total_loss

    def test_step(self, batch, batch_idx):
        results = self(batch)
        total_loss, loss_dict = self._compute_loss(results)

        # Log loss components
        actual_batch_size = len(batch['ligand'])
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':
                self.log(f"test_{loss_name}", loss_value, batch_size=actual_batch_size,
                        on_epoch=True, sync_dist=True)

        # Store predictions for epoch-level metrics
        pred_affinities = results['affinity'].detach().cpu().flatten()
        target_affinities = results['target_affinity'].detach().cpu().flatten()
        self.test_predictions.extend(pred_affinities.clone().tolist())
        self.test_targets.extend(target_affinities.clone().tolist())
        # self.test_predictions.extend(results['affinity'].detach().cpu().tolist())
        # self.test_targets.extend(results['target_affinity'].detach().cpu().tolist())

        # Compute additional metrics
        pred_affinity = results['affinity']
        affinities = results['target_affinity']
        mae = torch.mean(torch.abs(pred_affinity - affinities))

        # Log metrics
        self.log("test_mae", mae, prog_bar=True, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)

        # Log predictions for debugging
        self._log_predictions(batch, results['affinity'], results['target_affinity'], "Test")

        return total_loss

    # ========================================
    # EPOCH-LEVEL METRICS
    # ========================================

    def on_train_epoch_end(self):
        """Compute epoch-level training metrics"""
        if len(self.train_predictions) > 0:
            device = self.device if torch.cuda.is_available() else torch.device("cpu")
            predictions = torch.tensor(self.train_predictions, device=device)
            targets = torch.tensor(self.train_targets, device=device)

            # Compute correlation
            if len(predictions) > 1:
                pearson = torch.corrcoef(torch.stack([predictions, targets]))[0, 1]
                self.log("train_pearson", pearson, prog_bar=True, on_epoch=True, sync_dist=True)
            # Compute RMSE
            rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
            self.log("train_rmse", rmse, prog_bar=True, on_epoch=True, sync_dist=True)

            # Clear stored predictions
            self.train_predictions = []
            self.train_targets = []

    def on_validation_epoch_end(self):
        """Compute epoch-level validation metrics"""
        if len(self.val_predictions) > 0:
            device = self.device if torch.cuda.is_available() else torch.device("cpu")
            predictions = torch.tensor(self.val_predictions, device=device)
            targets = torch.tensor(self.val_targets, device=device)

            # Compute correlation
            if len(predictions) > 1:
                pearson = torch.corrcoef(torch.stack([predictions, targets]))[0, 1]
                self.log("val_pearson", pearson, prog_bar=True, on_epoch=True, sync_dist=True)
            # Compute RMSE
            rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
            self.log("val_rmse", rmse, prog_bar=True, on_epoch=True, sync_dist=True)
            # Clear stored predictions
            self.val_predictions = []
            self.val_targets = []

    def on_test_epoch_end(self):
        """Compute epoch-level test metrics"""
        if len(self.test_predictions) > 0:
            device = self.device if torch.cuda.is_available() else torch.device("cpu")
            predictions = torch.tensor(self.test_predictions, device=device)
            targets = torch.tensor(self.test_targets, device=device)
            # Compute correlation
            if len(predictions) > 1:
                pearson = torch.corrcoef(torch.stack([predictions, targets]))[0, 1]
                self.log("test_pearson", pearson, on_epoch=True, sync_dist=True)

            # Compute RMSE
            rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
            self.log("test_rmse", rmse, on_epoch=True, sync_dist=True)

            # Clear stored predictions
            self.test_predictions = []
            self.test_targets = []

    def configure_optimizers(self):
        # Use different learning rates for different components
        param_groups = [
            {'params': self.protein_virtual_encoder.parameters(), 'lr': self.hparams.lr * 1},
            {'params': self.ligand_encoder.parameters(), 'lr': self.hparams.lr * 1},
            {'params': self.regressor.parameters(), 'lr': self.hparams.lr},
        ]

        # Add structure predictor parameters if enabled
        if self.predict_str:
            param_groups.append({
                'params': self.structure_predictor.parameters(),
                'lr': self.hparams.lr
            })

        # Add other parameters
        other_params = []
        excluded_modules = ['protein_encoder','virtual_encoder','protein_virtual_encoder','ligand_encoder', 'regressor']
        if self.predict_str:
            excluded_modules.append('structure_predictor')

        for name, param in self.named_parameters():
            if not any(module in name for module in excluded_modules):
                other_params.append(param)

        if other_params:
            param_groups.append({'params': other_params, 'lr': self.hparams.lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Monitor total loss, not just reg_loss
                'interval': 'epoch',
                'frequency': 1
            }
        }

    # ========================================
    # ORIGINAL PROCESSING METHODS
    # ========================================

    def _process_protein_and_virtual_nodes_wo_init_encoder(self, protein_virtual_batch, protein_mask, virtual_mask):
        """Process protein and virtual nodes with GVP encoder"""
        """Process protein and virtual nodes without initial encoder"""
        node_s = protein_virtual_batch.node_s
        node_v = protein_virtual_batch.node_v
        edge_index = protein_virtual_batch.edge_index
        edge_s = protein_virtual_batch.edge_s
        edge_v = protein_virtual_batch.edge_v
        s, v = self.protein_virtual_encoder(node_s, node_v, edge_index, edge_s, edge_v)
        prot_s = s[protein_mask]
        prot_v = v[protein_mask]
        virtual_s = s[virtual_mask]
        virtual_v = v[virtual_mask]
        return prot_s, prot_v, virtual_s, virtual_v

    def _process_protein_and_virtual_nodes(self, protein_virtual_batch, protein_mask, virtual_mask):
        """Process protein and virtual nodes with GVP encoder"""
        node_s = protein_virtual_batch.node_s
        node_v = protein_virtual_batch.node_v
        edge_index = protein_virtual_batch.edge_index
        edge_s = protein_virtual_batch.edge_s
        edge_v = protein_virtual_batch.edge_v

        protein_indices_orig = protein_mask.nonzero(as_tuple=True)[0]
        virtual_indices_orig = virtual_mask.nonzero(as_tuple=True)[0]

        # --- Subgraph for Protein Nodes ---
        p_s = node_s[protein_mask]
        p_v = node_v[protein_mask]
        p2p_edge_mask = protein_mask[edge_index[0]] & protein_mask[edge_index[1]]
        p2p_edge_index = edge_index[:, p2p_edge_mask]
        p2p_edge_s = edge_s[p2p_edge_mask]
        p2p_edge_v = edge_v[p2p_edge_mask]

        # Remap protein subgraph edge_index to local indices
        p_node_map = torch.full((protein_virtual_batch.num_nodes,), -1, dtype=torch.long, device=edge_index.device)
        p_node_map[protein_indices_orig] = torch.arange(protein_indices_orig.size(0), device=edge_index.device)
        p2p_edge_index_remapped = p_node_map[p2p_edge_index]


        # --- Subgraph for Virtual Nodes ---
        v_s = node_s[virtual_mask]
        v_v = node_v[virtual_mask]
        v2v_edge_mask = virtual_mask[edge_index[0]] & virtual_mask[edge_index[1]]
        v2v_edge_index = edge_index[:, v2v_edge_mask]
        v2v_edge_s = edge_s[v2v_edge_mask]
        v2v_edge_v = edge_v[v2v_edge_mask]

        # Remap virtual subgraph edge_index to local indices
        v_node_map = torch.full((protein_virtual_batch.num_nodes,), -1, dtype=torch.long, device=edge_index.device)
        v_node_map[virtual_indices_orig] = torch.arange(virtual_indices_orig.size(0), device=edge_index.device)
        v2v_edge_index_remapped = v_node_map[v2v_edge_index]

        # Get initial embeddings for protein and virtual nodes using their respective encoders
        p_s_encoded, p_v_encoded = self.protein_encoder(p_s, p_v, p2p_edge_index_remapped, p2p_edge_s, p2p_edge_v)
        v_s_encoded, v_v_encoded = self.virtual_encoder(v_s, v_v, v2v_edge_index_remapped, v2v_edge_s, v2v_edge_v)

        # Combine protein and virtual node embeddings
        combined_node_s = torch.cat([p_s_encoded, v_s_encoded], dim=0)
        combined_node_v = torch.cat([p_v_encoded, v_v_encoded], dim=0)

        # Remap the global edge_index to the new combined node order
        num_protein_nodes_in_batch = protein_indices_orig.size(0)
        node_map = torch.full((protein_virtual_batch.num_nodes,), -1, dtype=torch.long, device=edge_index.device)
        node_map[protein_indices_orig] = torch.arange(num_protein_nodes_in_batch, device=edge_index.device)
        node_map[virtual_indices_orig] = torch.arange(virtual_indices_orig.size(0), device=edge_index.device) + num_protein_nodes_in_batch

        node_is_pv_mask = protein_mask | virtual_mask
        pv_edge_mask = node_is_pv_mask[edge_index[0]] & node_is_pv_mask[edge_index[1]]

        full_pv_edge_index = edge_index[:, pv_edge_mask]
        full_pv_edge_s = edge_s[pv_edge_mask]
        full_pv_edge_v = edge_v[pv_edge_mask]

        remapped_edge_index = node_map[full_pv_edge_index]

        # Process the combined graph with the correctly remapped edge index and attributes
        s_final, v_final = self.protein_virtual_encoder(
            combined_node_s, combined_node_v, remapped_edge_index, full_pv_edge_s, full_pv_edge_v
        )

        # Split the final embeddings back into protein and virtual parts.
        prot_s, virtual_s = torch.split(s_final, [num_protein_nodes_in_batch, s_final.size(0) - num_protein_nodes_in_batch], dim=0)
        prot_v, virtual_v = torch.split(v_final, [num_protein_nodes_in_batch, v_final.size(0) - num_protein_nodes_in_batch], dim=0)

        return prot_s, prot_v, virtual_s, virtual_v

    def _get_virtual_subgraph(self, protein_virtual_batch, virtual_mask):
        # this sub function was written with Gemini + bit of re-naming by repo owner
        # Get the indices of the virtual nodes in the large graph
        virtual_node_indices = virtual_mask.nonzero(as_tuple=True)[0]

        # Filter the edges to keep only those where both source and destination are virtual nodes
        v2v_edge_mask = virtual_mask[protein_virtual_batch.edge_index[0]] & virtual_mask[protein_virtual_batch.edge_index[1]]
        v2v_edge_index = protein_virtual_batch.edge_index[:, v2v_edge_mask]

        # Remap the edge indices to be in the range [0, num_virtual_nodes - 1]
        node_map = torch.full((protein_virtual_batch.num_nodes,), -1, dtype=torch.long, device=virtual_node_indices.device)
        node_map[virtual_node_indices] = torch.arange(virtual_node_indices.size(0), device=virtual_node_indices.device)
        v2v_edge_index_remapped = node_map[v2v_edge_index]

        return v2v_edge_index_remapped

    def _pool_virtual_nodes(self, s, coords, edge_index, batch_idx):

        if s.shape[0] == 0:
            return s, coords, batch_idx
        # print("Inside SLURM job:")
        # print("CUDA Available:", torch.cuda.is_available())
        # print("torch version:", torch.__version__)
        # print("device count:", torch.cuda.device_count())
        # print("device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
        # print('self.device:', self.device)
        # print('edge device:', edge_index.device)
        cluster = graclus(edge_index, num_nodes=s.size(0))

        unique_clusters, cluster_map = torch.unique(cluster, return_inverse=True)

        # Aggregate features and coordinates by averaging over each cluster
        s_pooled = scatter_mean(s, cluster_map, dim=0)
        coords_pooled = scatter_mean(coords, cluster_map, dim=0)
        batch_idx_pooled = scatter_mean(batch_idx.float(), cluster_map, dim=0).long()

        # num_before = s.shape[0]
        # num_after = s_pooled.shape[0]
        # self.print(f"Virtual node pooling: {num_before} -> {num_after} nodes.")

        return s_pooled, coords_pooled, batch_idx_pooled

    def _process_ligand_nodes(self, ligand_batch):
        """Process ligand nodes with GVP encoder"""
        lx_s = ligand_batch.x
        le_s = ligand_batch.edge_attr
        le_idx = ligand_batch.edge_index
        lx_v_dummy = torch.zeros(lx_s.size(0), 0, 3, device=lx_s.device)
        le_v_dummy = torch.zeros(le_s.size(0), 0, 3, device=le_s.device)
        lig_s, _ = self.ligand_encoder(lx_s, lx_v_dummy, le_idx, le_s, le_v_dummy)
        return lig_s


    def _log_predictions(self, batch, y_pred, y, stage):
        # Flatten tensors to ensure they're always 1D
        pred_list = y_pred.flatten().tolist()
        true_list = y.flatten().tolist()

        for i, (pred, true) in enumerate(zip(pred_list, true_list)):
            name = batch.get('name', [f"{stage}_sample_{i}"])[i] if 'name' in batch else f"{stage}_sample_{i}"
            self.print(f"[{stage}] {name}: True Aff = {true:.3f}, Predicted = {pred:.3f}")
