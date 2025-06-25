# in src/model/multi_three.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.bin
from torch_geometric.data import Batch
from torch_geometric.nn.pool import graclus
from torch_scatter import scatter_mean
from src.model.gvp_encoder import GVPGraphEncoderHybrid as GVPGraphEncoder
from src.model.my_modules import GridBasedStructModulePadded as GridBasedStructModule
from src.model.my_modules import PairwiseAttentionStructModule as StructModule
from src.model.my_modules import VectorizedPairwiseAttentionStructModule as StructModule_v2
from src.model.my_modules import DirectCoordinatePredictor as SimpleStructModule
from .loss_utils import MultiTaskAffinityCoordinateLoss
from .loss_utils import HuberReplacementLoss as AffinityLoss

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
                 loss_type="single",  # single: no classification head
                 loss_params=None):  
        super().__init__()
        self.save_hyperparameters()
        
        # === Original model components ===
        self.interaction_mode = interaction_mode
        self.predict_str = predict_str
        
        # Encoders
        self.protein_virtual_encoder = GVPGraphEncoder(
            node_dims=protein_node_dims,
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
            self.coord_loss_weight = 0.4 
            # self.structure_predictor = GridBasedStructModule(
            #         embed_dim=embed_dim, num_heads=4
            #     )
            self.structure_predictor = SimpleStructModule(
                lig_embed_dim=ligand_hidden_dims[0],
                prot_embed_dim=embed_dim,
                hidden_dim=128
            ) 
        else: 
            self.coord_loss_weight = 0.0

        # === Enhanced components ===
        # Main regression head
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
            # self.thresholds = [-3, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
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
            self.loss_fn = WeightedCategoryLoss_v2(
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
            self.loss_fn = AffinityLoss(
                delta=0.5,
                extreme_weight=1.6,
                ranking_weight=0.5) 
        
        if self.predict_str:
            # Create combined loss function
            self.combined_loss_fn = MultiTaskAffinityCoordinateLoss(
                affinity_loss_fn=self.loss_fn, 
                coord_loss_weight=self.coord_loss_weight,
                coord_loss_type="mse",
                lig_internal_dist_w=0 #0.1
            )

        self.loss_type = loss_type
    

    def forward(self, batch):
        interaction_results = self.get_embeddings(batch)
        complex_embedding = interaction_results['complex_embedding']  # [B, embed_dim]

        if self.loss_type == "multitask":
            # Regression prediction
            pred_affinity = self.regressor(complex_embedding).squeeze()
            # Classification prediction
            pred_logits = self.classification_head(complex_embedding)
            # Get true affinities
            affinities = batch['affinity'].to(self.device)
            results = {
                'affinity': pred_affinity,
                'logits': pred_logits,
                'target_affinity': affinities
            }
        else:
            # Single-task regression
            pred_affinity = self.regressor(complex_embedding).squeeze()
            affinities = batch['affinity'].to(self.device)
            results = {
                'affinity': pred_affinity,
                'target_affinity': affinities
            }

        if self.predict_str:
            coord_results = self._predict_coordinates_from_shared_embeddings(batch, interaction_results)
            if coord_results['predicted_ligand_coords'] is not None:
                results.update(coord_results)
        
        return results
    
    def get_embeddings(self, batch):
        """Get embeddings before final prediction (for multi-task learning)"""
        protein_virtual_batch = batch['protein_virtual'].to(self.device)
        ligand_batch = batch['ligand'].to(self.device)

        X_sidechain_padded = protein_virtual_batch.X_sidechain_padded.to(self.device)  # [B, N_prot, N_sidechain_max, 3]
        X_sidechain_mask = protein_virtual_batch.X_sidechain_mask.to(self.device)  # [B, N_prot, N_sidechain_max]
        res_list = protein_virtual_batch.res_list  # Residue list for reference
  
        # Split protein and water nodes
        protein_mask = protein_virtual_batch.node_type == 0
        # water_mask = protein_batch.node_type == 1
        virtual_mask = protein_virtual_batch.node_type == 2
        
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
            'res_list': res_list,  # Residue list for reference
            'X_sidechain_padded': X_sidechain_padded,  # [B, N_prot, N_sidechain_max, 3]
            'X_sidechain_mask': X_sidechain_mask,      # [B, N_prot, N_sidechain_max]
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
            # Only protein - need to add batch dimension properly  
            complex_repr = p.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, D]
            virtual_after_interaction = torch.empty(0, embed_dim, device=self.device)
        elif v.size(0) > 0:
            # Only water - need to add batch dimension properly
            complex_repr = v.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, D]
            virtual_after_interaction = v  # Use virtual node embeddings given as input
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
        ligand_embeddings = interaction_results['ligand_embeddings_f'] 
        ligand_batch_idx = interaction_results['ligand_batch_idx']

        ligand_batch = batch['ligand'].to(self.device)
        
        protein_embeddings = interaction_results['protein_embeddings_f']  # [N_protein, embed_dim]
        protein_batch_idx = interaction_results['protein_batch_idx']  # [N_protein]
        X_sidechain_padded = interaction_results['X_sidechain_padded']  # [N_prot, N_sidechain_max, 3]
        X_sidechain_mask = interaction_results['X_sidechain_mask']      # [N_prot, N_sidechain_max]

        # Create target mask for ligand atoms
        batch_size = ligand_batch_idx.max().item() + 1
        max_ligand_atoms = torch.bincount(ligand_batch_idx).max().item()
        max_protein_residues = torch.bincount(protein_batch_idx).max().item()

        target_mask = torch.zeros(batch_size, max_ligand_atoms, dtype=torch.bool, device=self.device)
        for b in range(batch_size):
            num_atoms = (ligand_batch_idx == b).sum().item()
            target_mask[b, :num_atoms] = True
        
        protein_mask = torch.zeros(batch_size, max_protein_residues, dtype=torch.bool, device=self.device)
        for b in range(batch_size):
            num_residues = (protein_batch_idx == b).sum().item()
            protein_mask[b, :num_residues] = True

        max_sidechain_atoms = X_sidechain_mask.size(1)
    
        X_sidechain_mask_batched = torch.zeros(
            batch_size, max_protein_residues, max_sidechain_atoms, 
            dtype=torch.bool, device=self.device
        )
        X_sidechain_coords_batched = torch.zeros(
        batch_size, max_protein_residues, max_sidechain_atoms, 3,
        device=self.device
        )   

        mask_idx = 0
        for b in range(batch_size):
            num_residues_in_batch = (protein_batch_idx == b).sum().item()
            if num_residues_in_batch > 0:
                batch_masks = X_sidechain_mask[mask_idx:mask_idx + num_residues_in_batch]
                X_sidechain_mask_batched[b, :num_residues_in_batch] = batch_masks

                batch_coords = X_sidechain_padded[mask_idx:mask_idx + num_residues_in_batch]
                X_sidechain_coords_batched[b, :num_residues_in_batch] = batch_coords

                mask_idx += num_residues_in_batch

        # Use the structure module with shared embeddings
        pred_ligand_coords, pred_sidechain_coords = self.structure_predictor(
            # virtual_embeddings=virtual_embeddings,
            # virtual_coords=virtual_coords, 
            # virtual_batch_idx=virtual_batch_idx,
            ligand_embeddings=ligand_embeddings,
            ligand_batch_idx=ligand_batch_idx,
            protein_embeddings=protein_embeddings,
            protein_batch_idx=protein_batch_idx,
            X_sidechain_mask=X_sidechain_mask_batched,
            target_mask=target_mask,
            protein_mask=protein_mask
        )

        # Get ground truth coordinates for ligand atoms
        target_lig_coords = torch.zeros_like(pred_ligand_coords)
        ligand_coords = ligand_batch.pos
        
        coord_idx = 0
        for b in range(batch_size):
            num_atoms = (ligand_batch_idx == b).sum().item()
            target_lig_coords[b, :num_atoms] = ligand_coords[coord_idx:coord_idx + num_atoms]
            coord_idx += num_atoms
        
        return {
            'predicted_ligand_coords': pred_ligand_coords,
            'target_ligand_coords': target_lig_coords,
            'predicted_sidechain_coords': pred_sidechain_coords,
            'target_sidechain_coords': X_sidechain_coords_batched,
            'lig_coord_target_mask': target_mask,
            'sidechain_mask': X_sidechain_mask_batched,  # [B, N_prot, N_sidechain_max]
            'prot_coord_target_mask': protein_mask
        }

    # ========================================
    # LOSS COMPUTATION AND TRAINING STEPS
    # ========================================

    def _compute_loss(self, results):
        """Unified loss computation for both affinity and coordinate prediction"""
        if self.predict_str and 'predicted_ligand_coords' in results:
            # Use combined loss function
            loss_kwargs = {
                'pred_affinity': results['affinity'],
                'target_affinity': results['target_affinity']
            }
            
            if self.loss_type == "multitask":
                loss_kwargs['pred_logits'] = results['logits']
            
            loss_kwargs.update({
                'pred_lig_coords': results['predicted_ligand_coords'],
                'target_lig_coords': results['target_ligand_coords'],
                'lig_coord_mask': results['lig_coord_target_mask'],
                'pred_sidechain_coords': results['predicted_sidechain_coords'],
                'target_sidechain_coords': results['target_sidechain_coords'],
                'prot_coord_mask': results['prot_coord_target_mask'],
                'sidechain_mask': results['sidechain_mask']

            })
            
            total_loss, loss_dict = self.combined_loss_fn(**loss_kwargs)
            return total_loss, loss_dict
        else:
            # Affinity-only loss
            if self.loss_type == "multitask":
                affinity_result = self.loss_fn(
                    results['affinity'], results['target_affinity'], results['logits']
                )
                if isinstance(affinity_result, tuple):
                    total_loss, reg_loss, cat_penalty, extreme_penalty, pearson_pen = affinity_result
                    loss_dict = {
                        'total_loss': total_loss,
                        'affinity_loss': total_loss,
                        'reg_loss': reg_loss,
                        'category_loss': cat_penalty,
                        'extreme_penalty': extreme_penalty,
                        'ranking_loss': pearson_pen # pearson penalty is used as ranking loss
                    }
            else:
                affinity_result = self.loss_fn(results['affinity'], results['target_affinity'])
                if isinstance(affinity_result, tuple):
                    total_loss, reg_loss, ranking_loss, _, _ = affinity_result
                    loss_dict = {'total_loss': total_loss,
                                 'affinity_loss': total_loss,
                                 'reg_loss': reg_loss, 
                                 'ranking_loss': ranking_loss}
            
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
        self.test_predictions.extend(pred_affinities.tolist())
        self.test_targets.extend(target_affinities.tolist())
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
        excluded_modules = ['protein_virtual_encoder','ligand_encoder', 'regressor']
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

    def _process_protein_and_virtual_nodes(self, protein_virtual_batch, protein_mask, virtual_mask):
        """Process protein and virtual nodes with GVP encoder"""
        node_s = protein_virtual_batch.node_s
        node_v = protein_virtual_batch.node_v
        edge_index = protein_virtual_batch.edge_index
        edge_s = protein_virtual_batch.edge_s
        edge_v = protein_virtual_batch.edge_v
        # process including water nodes
        s, v = self.protein_virtual_encoder(node_s, node_v, edge_index, edge_s, edge_v)
        # then just filter for protein and virtual nodes
        prot_s, prot_v = s[protein_mask], v[protein_mask]
        virtual_s, virtual_v  = s[virtual_mask], v[virtual_mask]

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