# in src/model/multi_three.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from src.model.gvp_encoder import GVPGraphEncoderHybrid as GVPGraphEncoder
from src.model.my_modules import GridBasedStructModule
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
        self.protein_encoder = GVPGraphEncoder(
            node_dims=protein_node_dims,
            edge_dims=protein_edge_dims,
            hidden_dims=protein_hidden_dims,
            num_layers=num_gvp_layers,
            drop_rate=dropout
        )
        # virtual nodes are padded to the same size as protein nodes
        self.virtual_encoder = GVPGraphEncoder(
            node_dims=virtual_node_dims, # = protein_node_dims
            edge_dims=protein_edge_dims,
            hidden_dims=virtual_hidden_dims,
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
        
        # Structure Prediction Setups
        if predict_str:
            self.coord_loss_weight = 0.3 
            self.structure_predictor = GridBasedStructModule(
                    embed_dim=embed_dim, num_heads=4
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
                lig_internal_dist=True
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
            results.update(coord_results)
        
        return results
    
    def get_embeddings(self, batch):
        """Get embeddings before final prediction (for multi-task learning)"""
        protein_batch = batch['protein_virtual'].to(self.device)
        ligand_batch = batch['ligand'].to(self.device)
        
        # Split protein and water nodes
        protein_mask = protein_batch.node_type == 0
        # water_mask = protein_batch.node_type == 1
        virtual_mask = protein_batch.node_type == 2
        
        # Step1: Process prot/lig component
        prot_s = self._process_protein_nodes(protein_batch, protein_mask)
        lig_s = self._process_ligand_nodes(ligand_batch)

        # Step2: Process water nodes (virtual nodes) using updated prot_s
        virtual_s= self._process_virtual_nodes(protein_batch, virtual_mask, prot_s, protein_mask)
        
        # Step3: Get interaction embeddings (before regression)
        complex_embeddings, virtual_embeddings_enhanced = self._get_hierarchical_interaction_embeddings(
            prot_s, virtual_s, lig_s, protein_batch, protein_mask, virtual_mask, ligand_batch
        )
        
        return {
            'complex_embedding': complex_embeddings,           # [B, embed_dim] for affinity
            'enhanced_virtual_embeddings': virtual_embeddings_enhanced,  # [N_virtual, embed_dim] for coords
            'virtual_batch_idx': protein_batch.batch[virtual_mask],      # Batch indices
            'virtual_coords': protein_batch.x[virtual_mask]             # Grid coordinates
        }      
    
    def _get_hierarchical_interaction_embeddings(self, prot_s, virtual_s, lig_s, 
                                                      protein_batch, protein_mask, virtual_mask, ligand_batch):
        """Get_embeddings-Step3: Hierarchical interaction but track virtual embeddings for coordinate prediction"""
        
        # Get batch information
        prot_batch_sizes = torch.bincount(protein_batch.batch[protein_mask]).tolist() if protein_mask.any() else []
        virtual_batch_sizes = torch.bincount(protein_batch.batch[virtual_mask]).tolist() if virtual_mask.any() else []
        lig_batch_sizes = torch.bincount(ligand_batch.batch).tolist()
        
        # Split by batch
        prot_s_list = torch.split(prot_s, prot_batch_sizes) if prot_s.size(0) > 0 else []
        virtual_s_list = torch.split(virtual_s, virtual_batch_sizes) if virtual_s.size(0) > 0 else []
        lig_s_list = torch.split(lig_s, lig_batch_sizes)
        
        complex_embeddings = []
        virtual_embeddings_after_interaction = []
        batch_size = len(lig_s_list)

        for i in range(batch_size):
            l = lig_s_list[i]
            p = prot_s_list[i] if i < len(prot_s_list) else torch.empty(0, self.hparams.protein_hidden_dims[0], device=self.device)
            v = virtual_s_list[i] if i < len(virtual_s_list) else torch.empty(0, self.hparams.protein_hidden_dims[0], device=self.device)
            
            # Hierarchical interaction with virtual tracking
            complex_emb, virtual_after_interaction = self._hierarchical_interaction_with_virtual(p, v, l)
            
            complex_embeddings.append(complex_emb)
            virtual_embeddings_after_interaction.append(virtual_after_interaction)
        
        # Stack results
        complex_embeddings = torch.stack(complex_embeddings)  # [B, embed_dim]
        
        # Concatenate virtual embeddings (maintaining batch order)
        if virtual_embeddings_after_interaction and all(v.size(0) > 0 for v in virtual_embeddings_after_interaction):
            virtual_embeddings_enhanced = torch.cat(virtual_embeddings_after_interaction, dim=0)
        else:
            virtual_embeddings_enhanced = virtual_s  # Fallback to input
        
        return complex_embeddings, virtual_embeddings_enhanced

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

        # Step 2: Complex-Ligand interaction
        if l.size(0) > 0:
            l_batch = l.unsqueeze(0)  # [1, N_l, D]
            
            # Ligand attends to protein-water complex
            final_out, _ = self.complex_ligand_attn(l_batch, complex_repr, complex_repr)  # [1, N_l, D]
            final_repr = final_out.mean(dim=1)  # [1, D]
        else:
            final_repr = complex_repr.squeeze(1)  # Remove sequence dimension: [1, D]

        return final_repr.squeeze(0), virtual_after_interaction # [D]

    def _predict_coordinates_from_shared_embeddings(self, batch, interaction_results):
        """Use enhanced virtual embeddings from shared pipeline for coordinate prediction"""
        virtual_embeddings = interaction_results['enhanced_virtual_embeddings']  # [N_virtual, embed_dim]
        virtual_coords = interaction_results['virtual_coords']                   # [N_virtual, 3]
        virtual_batch_idx = interaction_results['virtual_batch_idx']             # [N_virtual]
        
        ligand_batch = batch['ligand'].to(self.device)
        ligand_batch_idx = ligand_batch.batch
        
        # Create target mask for ligand atoms
        batch_size = ligand_batch_idx.max().item() + 1
        max_ligand_atoms = torch.bincount(ligand_batch_idx).max().item()
        
        target_mask = torch.zeros(batch_size, max_ligand_atoms, dtype=torch.bool, device=self.device)
        for b in range(batch_size):
            num_atoms = (ligand_batch_idx == b).sum().item()
            target_mask[b, :num_atoms] = True
        
        # Use the structure module with shared embeddings
        print ("[Debug] Using shared embeddings for structure prediction")
        print (f"[Debug] Virtual embeddings shape: {virtual_embeddings.shape}")
        print (f"[Debug] Virtual coords shape: {virtual_coords.shape}")
        print (f"[Debug] Target mask shape: {target_mask.shape}")
        print (f"[Debug] Virtual batch idx shape: {virtual_batch_idx.shape}")
        pred_coords, attention_weights = self.structure_predictor(
            virtual_embeddings, virtual_coords, target_mask, virtual_batch_idx
        )
        
        # Get ground truth coordinates
        target_coords = torch.zeros_like(pred_coords)
        ligand_coords = ligand_batch.pos
        
        coord_idx = 0
        for b in range(batch_size):
            num_atoms = (ligand_batch_idx == b).sum().item()
            target_coords[b, :num_atoms] = ligand_coords[coord_idx:coord_idx + num_atoms]
            coord_idx += num_atoms
        
        return {
            'predicted_ligand_coords': pred_coords,
            'target_ligand_coords': target_coords,
            'coord_attention_weights': attention_weights,
            'coord_target_mask': target_mask
        }

    def training_step(self, batch, batch_idx):
        results = self(batch)
        # Affinity loss
        if self.loss_type == "multitask":
            affinity_loss, reg_loss, cat_penalty, extreme_penalty, pearson_pen = self.loss_fn(
                results['affinity'], results['target_affinity'], results['logits']
            )
            self._log_predictions(batch, results['affinity'], results['target_affinity'], "Train")
            self.log("train_reg_loss", reg_loss, batch_size=actual_batch_size)
            self.log("train_cat_penalty", cat_penalty, batch_size=actual_batch_size)
            # self.log("train_extreme_penalty", extreme_penalty)
            self.log("train_Rs_penalty", pearson_pen, batch_size=actual_batch_size)
        else:
            affinity_loss = self.loss_fn(results['affinity'], results['target_affinity'])
        
        total_loss = affinity_loss


        elif self.loss_type == 'single':
            pred_affinity, affinities = self(batch)
            loss = self.loss_fn(pred_affinity, affinities)
            self._log_predictions(batch, pred_affinity, affinities, "Train")
        
        if self.coord_prediction and 'predicted_ligand_coords' in results:
            coord_loss = self._compute_coordinate_loss(results)
            loss += self.coord_loss_weight * coord_loss
            
            self.log("train_coord_loss", coord_loss, batch_size=len(batch['ligand']))
            
            # Log that we're using shared embeddings
            self.log("train_shared_embedding_coord_loss", coord_loss, batch_size=len(batch['ligand']))
        
        self.train_predictions.extend(pred_affinity.detach().cpu().tolist())
        self.train_targets.extend(affinities.detach().cpu().tolist())

        # Log main loss
        self.log("train_loss", loss, prog_bar=True, batch_size=actual_batch_size)
        
        return loss
    
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

    def validation_step(self, batch, batch_idx):
        actual_batch_size = len(batch['ligand']) if 'ligand' in batch else y.size(0)
        if self.loss_type == "multitask":
            pred_affinity, pred_logits, affinities = self(batch)
            loss, reg_loss, cat_penalty, extreme_penalty, pearson_pen = self.loss_fn(pred_affinity, affinities, pred_logits)

            # Log predictions
            self._log_predictions(batch, pred_affinity, affinities, "Valid")
            self.log("val_reg_loss", reg_loss, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)
            self.log("val_cat_penalty", cat_penalty, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)
            # self.log("val_extreme_penalty", extreme_penalty, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)
            self.log("val_Rs_penalty", pearson_pen, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)
        
        elif self.loss_type == 'single': 
            # Original (non-multitask) fallback
            pred_affinity, affinities = self(batch)
            loss = self.loss_fn(pred_affinity, affinities)
            self._log_predictions(batch, pred_affinity, affinities, "Valid")
        
        # Store predictions for epoch-level metrics
        self.val_predictions.extend(pred_affinity.detach().cpu().tolist())
        self.val_targets.extend(affinities.detach().cpu().tolist())
        
        # Compute additional metrics
        mae = torch.mean(torch.abs(pred_affinity - affinities))
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=actual_batch_size)
        self.log("val_mae", mae, prog_bar=True, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)
        
        return loss
    
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

    def test_step(self, batch, batch_idx):
        actual_batch_size = len(batch['ligand']) if 'ligand' in batch else y.size(0)
        if self.loss_type == "multitask":
            # Forward pass: get both affinity value and classification logits
            pred_affinity, pred_logits, affinities = self(batch)
            
            # Loss function (auto-generates class targets from affinities)
            total_loss, reg_loss, cat_penalty, extreme_penalty, pearson_pen = self.loss_fn(
                pred_affinity, affinities, pred_logits
            )

            # Log individual loss components
            self._log_predictions(batch, pred_affinity, affinities, "TEST")
            self.log("test_loss", total_loss, prog_bar=True, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)
            self.log("test_reg_loss", reg_loss, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)
            self.log("test_cat_penalty", cat_penalty, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)
            self.log("test_Rs_penalty", pearson_pen, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)

        elif self.loss_type == 'single':
            # Original (non-multitask) fallback
            # Forward pass: get affinity predictions
            pred_affinity, affinities = self(batch)
            total_loss = self.loss_fn(pred_affinity, affinities)
            self._log_predictions(batch, pred_affinity, affinities, "Test")
            self.log("test_reg_loss", reg_loss, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)
            
        # Track predictions for epoch-end analysis
        self.test_predictions.extend(pred_affinity.detach().cpu().tolist())
        self.test_targets.extend(affinities.detach().cpu().tolist())
        
        # Compute MAE
        mae = torch.mean(torch.abs(pred_affinity - affinities))
        self.log("test_mae", mae, prog_bar=True, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)
            
        return total_loss
        
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
            {'params': self.protein_encoder.parameters(), 'lr': self.hparams.lr * 1},
            {'params': self.virtual_encoder.parameters(), 'lr': self.hparams.lr * 1},
            {'params': self.ligand_encoder.parameters(), 'lr': self.hparams.lr * 1},
            {'params': self.regressor.parameters(), 'lr': self.hparams.lr},
        ]
        
        # Add other parameters
        other_params = []
        for name, param in self.named_parameters():
            if not any(module in name for module in ['protein_encoder', 'virtual_encoder', 'ligand_encoder', 'regressor']):
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
                'monitor': 'val_reg_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    # Include all the original methods (_process_protein_nodes, etc.) here

    def _process_protein_nodes(self, protein_batch, protein_mask):
        """Process protein nodes with GVP encoder"""
        if not protein_mask.any():
            return torch.empty(0, self.hparams.protein_hidden_dims[0], device=self.device)

        protein_node_indices = protein_mask.nonzero(as_tuple=True)[0]
        prot_x_s = protein_batch.node_s[protein_mask]
        prot_x_v = protein_batch.node_v[protein_mask]
        
        # Debug: Check shapes
        
        # Filter edges to only include protein-protein connections
        prot_edge_mask = protein_mask[protein_batch.edge_index[0]] & protein_mask[protein_batch.edge_index[1]]
        if not prot_edge_mask.any():
            # No protein-protein edges, use simple embedding
            print("No protein-protein edges found")
            return torch.zeros(prot_x_s.size(0), self.hparams.protein_hidden_dims[0], device=self.device)
            
        prot_edge_index = protein_batch.edge_index[:, prot_edge_mask]
        prot_e_s = protein_batch.edge_s[prot_edge_mask]
        prot_e_v = protein_batch.edge_v[prot_edge_mask]
        
        # Remap edge indices
        prot_node_mapping = torch.zeros(protein_batch.node_s.size(0), dtype=torch.long, device=self.device)
        prot_node_mapping[protein_node_indices] = torch.arange(len(protein_node_indices), device=self.device)
        prot_edge_index = prot_node_mapping[prot_edge_index]
        prot_s, _ = self.protein_encoder(prot_x_s, prot_x_v, prot_edge_index, prot_e_s, prot_e_v)
        return prot_s
    
    def _process_virtual_nodes(self, protein_batch, virtual_mask, prot_s, protein_mask):
        """Process virtual nodes with GVP encoder + protein context"""
        if not virtual_mask.any():
            # print("[Debug] No water nodes found in this sample")
            return torch.empty(0, self.hparams.protein_hidden_dims[0], device=self.device)

        virtual_node_indices = virtual_mask.nonzero(as_tuple=True)[0]
        virtual_x_s = protein_batch.node_s[virtual_mask]  # [N_water, 6]
        
        # Check if this sample actually has water nodes with proper features
        if virtual_x_s.size(0) == 0:
            # print("[Debug] Water mask found nodes but features are empty")
            return torch.empty(0, self.hparams.protein_hidden_dims[0], device=self.device)

        v_dim = max(1, self.hparams.virtual_node_dims[1])
        virtual_x_v = torch.zeros(virtual_x_s.size(0), v_dim, 3, device=self.device)
        # Process water with GVP encoder
        try:
            # combine (protein + virtual) => use only virtual
            virtual_s = self._encode_virtual_with_gvp(
                protein_batch, virtual_mask, virtual_node_indices, virtual_x_s, virtual_x_v
            )
            # print(f"[Debug] Water GVP encoding successful: {water_s.shape if water_s is not None else 'None'}")
        except Exception as e:
            # print(f"[Debug] Water GVP encoding failed: {e}")
            # Fallback: use simple embedding for water
            virtual_s = torch.zeros(virtual_x_s.size(0), self.hparams.protein_hidden_dims[0], device=self.device)
        
        # Handle None case
        if virtual_s is None:
            virtual_s = torch.zeros(virtual_x_s.size(0), self.hparams.protein_hidden_dims[0], device=self.device)
        
        # # Add protein context to water through cross-attention
        # if prot_s is not None and prot_s.size(0) > 0 and virtual_s.size(0) > 0:
        #     try:
        #         virtual_s = self._add_protein_context_to_water(
        #             virtual_s, prot_s, protein_batch, virtual_mask, protein_mask
        #         )
        #         # print(f"[Debug] Added protein context to water: {water_s.shape}")
        #     except Exception as e:
        #         pass  
        #         # print(f"[Debug] Failed to add protein context to water: {e}")
        
        return virtual_s

    def _encode_virtual_with_gvp(self, protein_batch, virtual_mask, virtual_node_indices, virtual_x_s, virtual_x_v):
        """Encode virtual nodes using GVP encoder with proper edge handling"""
        
        # Create edges for virtual nodes (v-v + v-protein interactions)
        virtual_edge_mask = virtual_mask[protein_batch.edge_index[0]] | virtual_mask[protein_batch.edge_index[1]]
        
        if not virtual_edge_mask.any():
            # No edges involving virtual node, return simple embedding
            return torch.zeros(virtual_x_s.size(0), self.hparams.water_hidden_dims[0], device=self.device)
        
        virtual_edge_index = protein_batch.edge_index[:, virtual_edge_mask]
        virtual_e_s = protein_batch.edge_s[virtual_edge_mask]
        virtual_e_v = protein_batch.edge_v[virtual_edge_mask]
        
        # Create mapping for all nodes (protein + water) to new indices
        all_nodes_mask = virtual_mask | (protein_batch.node_type == 0)  # protein or virtual
        all_node_indices = all_nodes_mask.nonzero(as_tuple=True)[0]
        node_mapping = torch.full((protein_batch.node_s.size(0),), -1, dtype=torch.long, device=self.device)
        node_mapping[all_node_indices] = torch.arange(len(all_node_indices), device=self.device)
        
        # Filter edges to only include valid node connections
        valid_edge_mask = (node_mapping[virtual_edge_index[0]] >= 0) & (node_mapping[virtual_edge_index[1]] >= 0)
        if not valid_edge_mask.any():
            return torch.zeros(virtual_x_s.size(0), self.hparams.water_hidden_dims[0], device=self.device)
            
        virtual_edge_index = virtual_edge_index[:, valid_edge_mask]
        virtual_e_s = virtual_e_s[valid_edge_mask]
        virtual_e_v = virtual_e_v[valid_edge_mask]
        virtual_edge_index = node_mapping[virtual_edge_index]
        
        # Combine protein and water features for the subgraph
        protein_indices = (protein_batch.node_type == 0) & all_nodes_mask
        
        # Get protein features
        prot_x_s_subgraph = protein_batch.node_s[protein_indices]
        prot_x_v_subgraph = protein_batch.node_v[protein_indices]
        
        # Ensure water vector features have the correct shape
        # where n_vector_dims should match protein vector dims
        if virtual_x_v.size(1) == 0 and prot_x_v_subgraph.size(1) > 0:
            # virtual has no vector features, but protein does - pad with zeros
            n_vector_dims = prot_x_v_subgraph.size(1)  # Should be 3 based on your node_v shape
            virtual_x_v = torch.zeros(virtual_x_s.size(0), n_vector_dims, 3, device=virtual_x_s.device)
        elif prot_x_v_subgraph.size(1) == 0 and virtual_x_v.size(1) > 0:
            # Protein has no vector features, but virtual does - pad protein with zeros
            n_vector_dims = virtual_x_v.size(1)
            prot_x_v_subgraph = torch.zeros(prot_x_s_subgraph.size(0), n_vector_dims, 3, device=prot_x_s_subgraph.device)
        assert prot_x_v_subgraph.shape[1] == virtual_x_v.shape[1], \
            f"Vector channel mismatch: protein={prot_x_v_subgraph.shape[1]}, virtual={virtual_x_v.shape[1]}"
        combined_x_s = torch.cat([prot_x_s_subgraph, virtual_x_s], dim=0)
        combined_x_v = torch.cat([prot_x_v_subgraph, virtual_x_v], dim=0)
        
        combined_s, combined_v = self.virtual_encoder(
            combined_x_s, combined_x_v, virtual_edge_index, virtual_e_s, virtual_e_v
        )
        
        # Extract only water features from the output
        num_protein_in_subgraph = protein_indices.sum().item()
        virtual_s = combined_s[num_protein_in_subgraph:]  # added sequentially / virtual features come after protein
        
        return virtual_s
    
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
        for i, (pred, true) in enumerate(zip(y_pred.tolist(), y.tolist())):
            # name = batch['ligand'][i].name if 'ligand' in batch and i < len(batch['ligand']) else f"{stage}_sample_{i}"
            name = batch['name'][i] if 'ligand_name' in batch else f"{stage}_sample_{i}"
            self.print(f"[{stage}] {name}: True Aff = {true:.3f}, Predicted = {pred:.3f}")
