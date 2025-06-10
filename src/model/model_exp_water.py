# in src/model/multi_three.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from src.model.gvp_encoder import GVPGraphEncoderHybrid as GVPGraphEncoder

class AFFModel_ThreeBody(pl.LightningModule):
    def __init__(self,
                 protein_node_dims=(26, 3),
                 water_node_dims=(26, 3),
                 protein_edge_dims=(32, 1),
                 ligand_node_dims=(46, 0),
                 ligand_edge_dims=(9, 0),
                 protein_hidden_dims=(196, 16),
                 water_hidden_dims=(196, 3),
                 ligand_hidden_dims=(196, 3),
                 num_gvp_layers=3,
                 dropout=0.1,
                 lr=1e-3,
                 interaction_mode="hierarchical",
                 loss_type="multitask",  # New parameter
                 loss_params=None):  # New parameter for loss configuration
        super().__init__()
        self.save_hyperparameters()
        
        # === Original model components ===
        self.interaction_mode = interaction_mode
        
        # Encoders
        self.protein_encoder = GVPGraphEncoder(
            node_dims=protein_node_dims,
            edge_dims=protein_edge_dims,
            hidden_dims=protein_hidden_dims,
            num_layers=num_gvp_layers,
            drop_rate=dropout
        )
        
        self.water_encoder = GVPGraphEncoder(
            node_dims=water_node_dims,
            edge_dims=protein_edge_dims,
            hidden_dims=water_hidden_dims,
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
            self.protein_water_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=4, batch_first=True
            )
            self.complex_ligand_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=4, batch_first=True
            )
            self.pw_fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim)
            )
        elif interaction_mode == "parallel":
            self.protein_ligand_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=4, batch_first=True
            )
            self.water_ligand_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=4, batch_first=True
            )
            self.protein_water_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=4, batch_first=True
            )
            self.three_way_fusion = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        
        self.water_context_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True
        )
        
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
        
        # Initialize loss function
        self._init_loss_function(loss_type, loss_params)
        
        # Additional heads for multi-task learning
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
            from .loss_utils import WeightedCategoryLoss_v2 as WeightedCategoryLoss
            # self.loss_fn = WeightedCategoryLoss(
            #         regression_weight=0.7,           # 70% weight on regression
            #         category_penalty_weight=0.2,     # 20% weight on category penalty
            #         extreme_penalty_weight=0.1,      # 10% weight on extreme preservation
            #         extreme_boost_low=1.1,           # 2x boost for < 4.0
            #         extreme_boost_high=1.1          # 1.5x boost for > 9.0
            #     )
            self.loss_fn = WeightedCategoryLoss(
                    thresholds=[-3, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
                    regression_weight=0.75,
                    category_penalty_weight=0.15,
                    extreme_penalty_weight=0,
                    pearson_penalty_weight=0.1,      # Optional
                    relative_error_weight=0.1,       # Optional
                    extreme_boost_low=1.3,
                    extreme_boost_high=1.4
                )
        elif loss_type == "single":
            self.loss_fn = nn.MSELoss()
        
        self.loss_type = loss_type
    
    def get_embeddings(self, batch):
        """Get embeddings before final prediction (for multi-task learning)"""
        protein_batch = batch['protein_water'].to(self.device)
        ligand_batch = batch['ligand'].to(self.device)
        
        # Split protein and water nodes
        protein_mask = protein_batch.node_type == 0
        water_mask = protein_batch.node_type == 1
        virtual_mask = protein_batch.node_type == 2
        
        # Process each component
        prot_s = self._process_protein_nodes(protein_batch, protein_mask)
        water_s = self._process_water_nodes(protein_batch, water_mask, prot_s, protein_mask)
        lig_s = self._process_ligand_nodes(ligand_batch)
        
        # Get interaction embeddings (before regression)
        embeddings = self._compute_three_body_embeddings(
            prot_s, water_s, lig_s, protein_batch, protein_mask, water_mask, ligand_batch
        )
        
        return embeddings
    
    def forward(self, batch):
        if self.loss_type == "multitask":
            # Get embeddings
            embeddings = self.get_embeddings(batch)
            
            # Regression prediction
            pred_affinity = self.regressor(embeddings).squeeze()
            
            # Classification prediction
            pred_logits = self.classification_head(embeddings)
            
            # Get true affinities
            affinities = batch['affinity'].to(self.device)
            
            return pred_affinity, pred_logits, affinities
        else:
            embeddings = self.get_embeddings(batch)
            pred_affinity = self.regressor(embeddings).squeeze()
            affinities = batch['affinity'].to(self.device)
            return pred_affinity, affinities
            
    
    def _compute_three_body_embeddings(self, prot_s, water_s, lig_s, protein_batch, 
                                      protein_mask, water_mask, ligand_batch):
        """Get embeddings for each sample (used in multi-task learning)"""
        # Similar to _compute_three_body_interactions but returns embeddings
        prot_batch_sizes = torch.bincount(protein_batch.batch[protein_mask]).tolist() if protein_mask.any() else []
        water_batch_sizes = torch.bincount(protein_batch.batch[water_mask]).tolist() if water_mask.any() else []
        lig_batch_sizes = torch.bincount(ligand_batch.batch).tolist()
        
        prot_s_list = torch.split(prot_s, prot_batch_sizes) if prot_s.size(0) > 0 else []
        water_s_list = torch.split(water_s, water_batch_sizes) if water_s.size(0) > 0 else []
        lig_s_list = torch.split(lig_s, lig_batch_sizes)
        
        embeddings = []
        batch_size = len(lig_s_list)
        # print (water_mask)
        # print (water_s)
        # print (water_s_list) 

        for i in range(batch_size):
            l = lig_s_list[i]
            p = prot_s_list[i] if i < len(prot_s_list) else torch.empty(0, self.hparams.protein_hidden_dims[0], device=self.device)
            w = water_s_list[i] if i < len(water_s_list) else torch.empty(0, self.hparams.protein_hidden_dims[0], device=self.device)
            # print (w)   
            if self.interaction_mode == "hierarchical":
                embedding = self._hierarchical_interaction(p, w, l)
            else:
                embedding = self._parallel_interaction(p, w, l)
            
            embeddings.append(embedding)
        
        return torch.stack(embeddings)
    
    def training_step(self, batch, batch_idx):
        actual_batch_size = len(batch['ligand']) if 'ligand' in batch else y.size(0)
        if self.loss_type == "multitask":
            pred_affinity, pred_logits, affinities = self(batch)
            loss, reg_loss, cat_penalty, extreme_penalty, pearson_pen = self.loss_fn(pred_affinity, affinities, pred_logits)

            # Log predictions
            self._log_predictions(batch, pred_affinity, affinities, "Train")
            self.log("train_reg_loss", reg_loss, batch_size=actual_batch_size)
            self.log("train_cat_penalty", cat_penalty, batch_size=actual_batch_size)
            # self.log("train_extreme_penalty", extreme_penalty)
            self.log("train_Rs_penalty", pearson_pen, batch_size=actual_batch_size)

        elif self.loss_type == 'single':
            pred_affinity, affinities = self(batch)
            loss = self.loss_fn(pred_affinity, affinities)
            self._log_predictions(batch, pred_affinity, affinities, "Train")
        
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
            {'params': self.protein_encoder.parameters(), 'lr': self.hparams.lr * 0.5},
            {'params': self.water_encoder.parameters(), 'lr': self.hparams.lr * 0.5},
            {'params': self.ligand_encoder.parameters(), 'lr': self.hparams.lr * 0.5},
            {'params': self.regressor.parameters(), 'lr': self.hparams.lr},
        ]
        
        # Add other parameters
        other_params = []
        for name, param in self.named_parameters():
            if not any(module in name for module in ['protein_encoder', 'water_encoder', 'ligand_encoder', 'regressor']):
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
    # They remain the same as in your original code

    def _process_protein_nodes(self, protein_batch, protein_mask):
        """Process protein nodes with GVP encoder"""
        if not protein_mask.any():
            return torch.empty(0, self.hparams.protein_hidden_dims[0], device=self.device)

        protein_node_indices = protein_mask.nonzero(as_tuple=True)[0]
        prot_x_s = protein_batch.node_s[protein_mask]
        prot_x_v = protein_batch.node_v[protein_mask]
        
        # Debug: Check shapes
        # print(f"Protein processing - prot_x_s: {prot_x_s.shape}, prot_x_v: {prot_x_v.shape}")
        
        # Filter edges to only include protein-protein connections
        prot_edge_mask = protein_mask[protein_batch.edge_index[0]] & protein_mask[protein_batch.edge_index[1]]
        if not prot_edge_mask.any():
            # No protein-protein edges, use simple embedding
            print("No protein-protein edges found")
            return torch.zeros(prot_x_s.size(0), self.hparams.protein_hidden_dims[0], device=self.device)
            
        prot_edge_index = protein_batch.edge_index[:, prot_edge_mask]
        prot_e_s = protein_batch.edge_s[prot_edge_mask]
        prot_e_v = protein_batch.edge_v[prot_edge_mask]
        
        # Debug: Check edge shapes
        # print(f"Protein edges - prot_e_s: {prot_e_s.shape}, prot_e_v: {prot_e_v.shape}")
        # print(f"Expected edge_s: (E, 32), edge_v: (E, 1, 3)")
        
        # Remap edge indices
        prot_node_mapping = torch.zeros(protein_batch.node_s.size(0), dtype=torch.long, device=self.device)
        prot_node_mapping[protein_node_indices] = torch.arange(len(protein_node_indices), device=self.device)
        prot_edge_index = prot_node_mapping[prot_edge_index]
        # Fix edge vector features shape (if needed)
        print (f"Processed protein edges - prot_edge_index: {prot_edge_index.shape}, prot_e_s: {prot_e_s.shape}, prot_e_v: {prot_e_v.shape}")
        print (f"Processed protein nodes - prot_x_s: {prot_x_s.shape}, prot_x_v: {prot_x_v.shape}")
        prot_s, _ = self.protein_encoder(prot_x_s, prot_x_v, prot_edge_index, prot_e_s, prot_e_v)
        return prot_s
    
    def _process_water_nodes(self, protein_batch, water_mask, prot_s, protein_mask):
        """Process water nodes with GVP encoder + protein context"""
        if not water_mask.any():
            # print("[Debug] No water nodes found in this sample")
            return torch.empty(0, self.hparams.protein_hidden_dims[0], device=self.device)

        water_node_indices = water_mask.nonzero(as_tuple=True)[0]
        water_x_s = protein_batch.node_s[water_mask]  # [N_water, 6]
        
        # print(f"[Debug] Processing {water_mask.sum()} water nodes")
        
        # Check if this sample actually has water nodes with proper features
        if water_x_s.size(0) == 0:
            # print("[Debug] Water mask found nodes but features are empty")
            return torch.empty(0, self.hparams.protein_hidden_dims[0], device=self.device)
        
        # Currently no vector features for water, but ready for dipole vectors
        # Future: water_x_v = protein_batch.node_v[water_mask]  # [N_water, 1, 3] for dipoles
        v_dim = max(1, self.hparams.water_node_dims[1])
        water_x_v = torch.zeros(water_x_s.size(0), v_dim, 3, device=self.device)
        # water_x_v = torch.zeros(water_x_s.size(0), self.hparams.water_node_dims[1], 3, device=self.device)
        # print("WATER x_v shape:", water_x_v.shape) 
        # Process water with GVP encoder
        try:
            water_s = self._encode_water_with_gvp(
                protein_batch, water_mask, water_node_indices, water_x_s, water_x_v
            )
            # print(f"[Debug] Water GVP encoding successful: {water_s.shape if water_s is not None else 'None'}")
        except Exception as e:
            # print(f"[Debug] Water GVP encoding failed: {e}")
            # Fallback: use simple embedding for water
            water_s = torch.zeros(water_x_s.size(0), self.hparams.protein_hidden_dims[0], device=self.device)
            # print(f"[Debug] Using fallback water embedding: {water_s.shape}")
        
        # Handle None case
        if water_s is None:
            # print("[Debug] Water_s is None, using fallback")
            water_s = torch.zeros(water_x_s.size(0), self.hparams.protein_hidden_dims[0], device=self.device)
        
        # Add protein context to water through cross-attention
        if prot_s is not None and prot_s.size(0) > 0 and water_s.size(0) > 0:
            try:
                water_s = self._add_protein_context_to_water(
                    water_s, prot_s, protein_batch, water_mask, protein_mask
                )
                # print(f"[Debug] Added protein context to water: {water_s.shape}")
            except Exception as e:
                pass  
                # print(f"[Debug] Failed to add protein context to water: {e}")
        
        return water_s

    def _encode_water_with_gvp(self, protein_batch, water_mask, water_node_indices, water_x_s, water_x_v):
        # print ("[Debug] Encoding water nodes with GVP encoder")
        """Encode water nodes using GVP encoder with proper edge handling"""
        
        # Create edges for water nodes (water-water + water-protein interactions)
        # Option 1: Include all edges involving water (recommended for realistic interactions)
        water_edge_mask = water_mask[protein_batch.edge_index[0]] | water_mask[protein_batch.edge_index[1]]
        # print(f"[Debug] Water edge mask: {water_edge_mask.sum()} edges involving water")
        # Option 2: Only water-water edges (uncomment if you want pure water-water interactions)
        # water_edge_mask = water_mask[protein_batch.edge_index[0]] & water_mask[protein_batch.edge_index[1]]
        
        if not water_edge_mask.any():
            # No edges involving water, return simple embedding
            # This is a fallback - in practice, water should have some edges
            return torch.zeros(water_x_s.size(0), self.hparams.water_hidden_dims[0], device=self.device)
        
        water_edge_index = protein_batch.edge_index[:, water_edge_mask]
        water_e_s = protein_batch.edge_s[water_edge_mask]
        water_e_v = protein_batch.edge_v[water_edge_mask]
        
        # Create mapping for all nodes (protein + water) to new indices
        all_nodes_mask = water_mask | (protein_batch.node_type == 0)  # protein or water
        all_node_indices = all_nodes_mask.nonzero(as_tuple=True)[0]
        node_mapping = torch.full((protein_batch.node_s.size(0),), -1, dtype=torch.long, device=self.device)
        node_mapping[all_node_indices] = torch.arange(len(all_node_indices), device=self.device)
        
        # Filter edges to only include valid node connections
        valid_edge_mask = (node_mapping[water_edge_index[0]] >= 0) & (node_mapping[water_edge_index[1]] >= 0)
        if not valid_edge_mask.any():
            return torch.zeros(water_x_s.size(0), self.hparams.water_hidden_dims[0], device=self.device)
            
        water_edge_index = water_edge_index[:, valid_edge_mask]
        water_e_s = water_e_s[valid_edge_mask]
        water_e_v = water_e_v[valid_edge_mask]
        water_edge_index = node_mapping[water_edge_index]
        
        # Combine protein and water features for the subgraph
        protein_indices = (protein_batch.node_type == 0) & all_nodes_mask
        
        # Get protein features
        prot_x_s_subgraph = protein_batch.node_s[protein_indices]
        prot_x_v_subgraph = protein_batch.node_v[protein_indices]
        
        # Ensure water vector features have the correct shape
        # Water currently has shape [N_water, 0, 3], but we need [N_water, n_vector_dims, 3]
        # where n_vector_dims should match protein vector dims
        if water_x_v.size(1) == 0 and prot_x_v_subgraph.size(1) > 0:
            # Water has no vector features, but protein does - pad with zeros
            n_vector_dims = prot_x_v_subgraph.size(1)  # Should be 3 based on your node_v shape
            water_x_v = torch.zeros(water_x_s.size(0), n_vector_dims, 3, device=water_x_s.device)
        elif prot_x_v_subgraph.size(1) == 0 and water_x_v.size(1) > 0:
            # Protein has no vector features, but water does - pad protein with zeros
            n_vector_dims = water_x_v.size(1)
            prot_x_v_subgraph = torch.zeros(prot_x_s_subgraph.size(0), n_vector_dims, 3, device=prot_x_s_subgraph.device)
        assert prot_x_v_subgraph.shape[1] == water_x_v.shape[1], \
            f"Vector channel mismatch: protein={prot_x_v_subgraph.shape[1]}, water={water_x_v.shape[1]}"
        combined_x_s = torch.cat([prot_x_s_subgraph, water_x_s], dim=0)
        combined_x_v = torch.cat([prot_x_v_subgraph, water_x_v], dim=0)
        
        # Run GVP encoder on the combined subgraph
        # print (f"[Debug] Combined subgraph - protein nodes: {prot_x_s_subgraph.shape}, water nodes: {water_x_s.shape}")
        # print (f"[Debug] Combined edge index: {water_edge_index.shape}, edge_s: {water_e_s.shape}, edge_v: {water_e_v.shape}")
        # print (f"[Debug] Combined node features - x_s: {combined_x_s.shape}, x_v: {combined_x_v.shape}")
        # print (f"[Debug] Combined node features - x_s: {combined_x_s.shape}, x_v: {combined_x_v.shape}")
        combined_s, combined_v = self.water_encoder(
            combined_x_s, combined_x_v, water_edge_index, water_e_s, water_e_v
        )
        
        # Extract only water features from the output
        num_protein_in_subgraph = protein_indices.sum().item()
        water_s = combined_s[num_protein_in_subgraph:]  # Water features come after protein
        
        return water_s

    def _add_protein_context_to_water(self, water_s, prot_s, protein_batch, water_mask, protein_mask):
        """Add protein context to water through cross-attention"""
        water_batch_idx = protein_batch.batch[water_mask]
        prot_batch_idx = protein_batch.batch[protein_mask]
        
        # Process batch by batch
        water_s_contextualized = []
        for batch_id in torch.unique(water_batch_idx):
            water_batch_mask = water_batch_idx == batch_id
            prot_batch_mask = prot_batch_idx == batch_id
            
            if water_batch_mask.any() and prot_batch_mask.any():
                w_batch = water_s[water_batch_mask].unsqueeze(0)  # [1, N_w, D]
                p_batch = prot_s[prot_batch_mask].unsqueeze(0)    # [1, N_p, D]
                
                # Water queries, protein keys/values
                w_contextualized, _ = self.water_context_attn(w_batch, p_batch, p_batch)
                water_s_contextualized.append(w_contextualized.squeeze(0))
            else:
                # No protein context available
                water_s_contextualized.append(water_s[water_batch_mask])
        
        if water_s_contextualized:
            water_s = torch.cat(water_s_contextualized, dim=0)
            
        return water_s

    def _process_ligand_nodes(self, ligand_batch):
        """Process ligand nodes with GVP encoder"""
        lx_s = ligand_batch.x
        le_s = ligand_batch.edge_attr
        le_idx = ligand_batch.edge_index
        lx_v_dummy = torch.zeros(lx_s.size(0), 0, 3, device=lx_s.device)
        le_v_dummy = torch.zeros(le_s.size(0), 0, 3, device=le_s.device)
        lig_s, _ = self.ligand_encoder(lx_s, lx_v_dummy, le_idx, le_s, le_v_dummy)
        return lig_s

    def _hierarchical_interaction(self, p, w, l):
        """Hierarchical: (P+W) -> complex -> interact with L"""
        embed_dim = self.hparams.protein_hidden_dims[0]
        
        # print(f"[Debug] Hierarchical interaction - P: {p.shape}, W: {w.shape}, L: {l.shape}")
        
        # Step 1: Protein-Water interaction (if both exist)
        
        if p.size(0) > 0 and w.size(0) > 0:
            # print("[Debug] Both protein and water present - computing P-W interaction")
            # Cross-attention between protein and water
            p_batch = p.unsqueeze(0)  # [1, N_p, D]
            w_batch = w.unsqueeze(0)  # [1, N_w, D]
            
            # Water attends to protein, protein attends to water
            w_enhanced, _ = self.protein_water_attn(w_batch, p_batch, p_batch)  # [1, N_w, D]
            p_enhanced, _ = self.protein_water_attn(p_batch, w_batch, w_batch)  # [1, N_p, D]
            
            # Pool and fuse
            p_pooled = p_enhanced.mean(dim=1)  # [1, D]
            w_pooled = w_enhanced.mean(dim=1)  # [1, D]
            
            pw_combined = torch.cat([p_pooled, w_pooled], dim=-1)  # [1, 2*D]
            complex_repr = self.pw_fusion(pw_combined).unsqueeze(1)  # [1, 1, D]
            
        elif p.size(0) > 0:
            # print("[Debug] Only protein present - using protein only")
            # Only protein - need to add batch dimension properly  
            complex_repr = p.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, D]
        elif w.size(0) > 0:
            # print("[Debug] Only water present - using water only")
            # Only water - need to add batch dimension properly
            complex_repr = w.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, D]
        else:
            # print("[Debug] Neither protein nor water present - using zero embedding")
            # Neither protein nor water - create properly shaped zero tensor
            complex_repr = torch.zeros(1, 1, embed_dim, device=self.device)  # [1, 1, D]

        # Step 2: Complex-Ligand interaction
        if l.size(0) > 0:
            l_batch = l.unsqueeze(0)  # [1, N_l, D]
            
            # Ligand attends to protein-water complex
            final_out, _ = self.complex_ligand_attn(l_batch, complex_repr, complex_repr)  # [1, N_l, D]
            final_repr = final_out.mean(dim=1)  # [1, D]
            # print(f"[Debug] Final representation with ligand: {final_repr.shape}")
        else:
            # print("[Debug] No ligand present - using complex only")
            final_repr = complex_repr.squeeze(1)  # Remove sequence dimension: [1, D]

        return final_repr.squeeze(0)  # [D]

    def _parallel_interaction(self, p, w, l):
        """Parallel: P-L, W-L, P-W interactions computed separately then fused"""
        embed_dim = self.hparams.protein_hidden_dims[0]
        
        # Initialize interaction representations
        pl_repr = torch.zeros(embed_dim, device=self.device)
        wl_repr = torch.zeros(embed_dim, device=self.device)
        pw_repr = torch.zeros(embed_dim, device=self.device)

        # Protein-Ligand interaction
        if p.size(0) > 0 and l.size(0) > 0:
            p_batch = p.unsqueeze(0)  # [1, N_p, D]
            l_batch = l.unsqueeze(0)  # [1, N_l, D]
            pl_out, _ = self.protein_ligand_attn(l_batch, p_batch, p_batch)  # [1, N_l, D]
            pl_repr = pl_out.mean(dim=1).squeeze(0)  # [D]

        # Water-Ligand interaction
        if w.size(0) > 0 and l.size(0) > 0:
            w_batch = w.unsqueeze(0)  # [1, N_w, D]
            l_batch = l.unsqueeze(0)  # [1, N_l, D]
            wl_out, _ = self.water_ligand_attn(l_batch, w_batch, w_batch)  # [1, N_l, D]
            wl_repr = wl_out.mean(dim=1).squeeze(0)  # [D]

        # Protein-Water interaction
        if p.size(0) > 0 and w.size(0) > 0:
            p_batch = p.unsqueeze(0)  # [1, N_p, D]
            w_batch = w.unsqueeze(0)  # [1, N_w, D]
            pw_out, _ = self.protein_water_attn(w_batch, p_batch, p_batch)  # [1, N_w, D]
            pw_repr = pw_out.mean(dim=1).squeeze(0)  # [D]

        # Fuse all three interactions
        three_way_concat = torch.cat([pl_repr, wl_repr, pw_repr], dim=-1)  # [3*D]
        final_repr = self.three_way_fusion(three_way_concat)  # [D]

        return final_repr
    
    def _log_predictions(self, batch, y_pred, y, stage):
        for i, (pred, true) in enumerate(zip(y_pred.tolist(), y.tolist())):
            # name = batch['ligand'][i].name if 'ligand' in batch and i < len(batch['ligand']) else f"{stage}_sample_{i}"
            name = batch['name'][i] if 'ligand_name' in batch else f"{stage}_sample_{i}"
            self.print(f"[{stage}] {name}: True Aff = {true:.3f}, Predicted = {pred:.3f}")
