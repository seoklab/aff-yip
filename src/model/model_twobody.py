# in src/model/multi_two.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from src.model.gvp_encoder import GVPGraphEncoderHybrid as GVPGraphEncoder

class AFFModel_TwoBody(pl.LightningModule):
    def __init__(self,
                 protein_node_dims=(6, 3),
                 protein_edge_dims=(32, 1),
                 ligand_node_dims=(46, 0),
                 ligand_edge_dims=(9, 0),
                 protein_hidden_dims=(196, 16),
                 ligand_hidden_dims=(196, 3),
                 num_gvp_layers=3,
                 dropout=0.1,
                 lr=1e-3,
                 interaction_mode="cross_attention",  # "cross_attention" or "concatenation"
                 loss_type="multitask",
                 loss_params=None):
        super().__init__()
        self.save_hyperparameters()
        
        # === Model components ===
        self.interaction_mode = interaction_mode
        
        # Encoders
        self.protein_encoder = GVPGraphEncoder(
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
        
        # Two-Body Interaction Modules
        embed_dim = protein_hidden_dims[0]
        
        if interaction_mode == "cross_attention":
            # Cross-attention between protein and ligand
            self.protein_ligand_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=4, batch_first=True
            )
            self.ligand_protein_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=4, batch_first=True
            )
            # Fusion layer for attended representations
            self.pl_fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim)
            )
        elif interaction_mode == "concatenation":
            # Simple concatenation and fusion
            self.pl_fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        
        # === Output heads ===
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
            self.classification_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 12)
            )
        
        # Track metrics
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
            self.loss_fn = WeightedCategoryLoss(
                    regression_weight=0.75,
                    category_penalty_weight=0.15,
                    extreme_penalty_weight=0,
                    pearson_penalty_weight=0.05,
                    relative_error_weight=0.05,
                    extreme_boost_low=1.0,
                    extreme_boost_high=1.4
                )
        elif loss_type == "single":
            self.loss_fn = nn.MSELoss()
        
        self.loss_type = loss_type
    
    def get_embeddings(self, batch):
        """Get embeddings before final prediction (for multi-task learning)"""
        protein_batch = batch['protein_water'].to(self.device)  # Changed from 'protein_water'
        ligand_batch = batch['ligand'].to(self.device)
        
        # Process each component
        prot_s = self._process_protein_nodes(protein_batch)
        lig_s = self._process_ligand_nodes(ligand_batch)
        
        # Get interaction embeddings (before regression)
        embeddings = self._compute_two_body_embeddings(
            prot_s, lig_s, protein_batch, ligand_batch
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
    
    def _compute_two_body_embeddings(self, prot_s, lig_s, protein_batch, ligand_batch):
        """Get embeddings for each sample (two-body interaction)"""
        prot_batch_sizes = torch.bincount(protein_batch.batch).tolist() if prot_s.size(0) > 0 else []
        lig_batch_sizes = torch.bincount(ligand_batch.batch).tolist()
        
        prot_s_list = torch.split(prot_s, prot_batch_sizes) if prot_s.size(0) > 0 else []
        lig_s_list = torch.split(lig_s, lig_batch_sizes)
        
        embeddings = []
        batch_size = len(lig_s_list)

        for i in range(batch_size):
            l = lig_s_list[i]
            p = prot_s_list[i] if i < len(prot_s_list) else torch.empty(0, self.hparams.protein_hidden_dims[0], device=self.device)
            
            if self.interaction_mode == "cross_attention":
                embedding = self._cross_attention_interaction(p, l)
            elif self.interaction_mode == "concatenation":
                embedding = self._concatenation_interaction(p, l)
            
            embeddings.append(embedding)
        
        return torch.stack(embeddings)
    
    def training_step(self, batch, batch_idx):
        actual_batch_size = len(batch['ligand']) if 'ligand' in batch else batch['affinity'].size(0)
        if self.loss_type == "multitask":
            pred_affinity, pred_logits, affinities = self(batch)
            loss, reg_loss, cat_penalty, extreme_penalty, pearson_pen = self.loss_fn(pred_affinity, affinities, pred_logits)

            # Log predictions
            self._log_predictions(batch, pred_affinity, affinities, "Train")
            self.log("train_reg_loss", reg_loss, batch_size=actual_batch_size)
            self.log("train_cat_penalty", cat_penalty, batch_size=actual_batch_size)
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
        actual_batch_size = len(batch['ligand']) if 'ligand' in batch else batch['affinity'].size(0)
        if self.loss_type == "multitask":
            pred_affinity, pred_logits, affinities = self(batch)
            loss, reg_loss, cat_penalty, extreme_penalty, pearson_pen = self.loss_fn(pred_affinity, affinities, pred_logits)

            # Log predictions
            self._log_predictions(batch, pred_affinity, affinities, "Valid")
            self.log("val_reg_loss", reg_loss, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)
            self.log("val_cat_penalty", cat_penalty, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)
            self.log("val_Rs_penalty", pearson_pen, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)
        
        elif self.loss_type == 'single': 
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
        actual_batch_size = len(batch['ligand']) if 'ligand' in batch else batch['affinity'].size(0)
        if self.loss_type == "multitask":
            pred_affinity, pred_logits, affinities = self(batch)
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
            pred_affinity, affinities = self(batch)
            total_loss = self.loss_fn(pred_affinity, affinities)
            self._log_predictions(batch, pred_affinity, affinities, "Test")
            self.log("test_loss", total_loss, prog_bar=True, batch_size=actual_batch_size, on_epoch=True, sync_dist=True)
            
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
            {'params': self.ligand_encoder.parameters(), 'lr': self.hparams.lr * 0.5},
            {'params': self.regressor.parameters(), 'lr': self.hparams.lr},
        ]
        
        # Add other parameters
        other_params = []
        for name, param in self.named_parameters():
            if not any(module in name for module in ['protein_encoder', 'ligand_encoder', 'regressor']):
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
                'monitor': 'val_loss' if self.loss_type == 'single' else 'val_reg_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def _process_protein_nodes(self, protein_batch):
        """Process protein nodes with GVP encoder"""
        prot_x_s = protein_batch.node_s
        prot_x_v = protein_batch.node_v
        prot_edge_index = protein_batch.edge_index
        prot_e_s = protein_batch.edge_s
        prot_e_v = protein_batch.edge_v
        
        prot_s, _ = self.protein_encoder(prot_x_s, prot_x_v, prot_edge_index, prot_e_s, prot_e_v)
        return prot_s
    
    def _verify_and_fix_edge_shapes(self, edge_s, edge_v, expected_edge_dims):
        """
        Ensure edge_v has shape [E, expected_vector_dim, 3] for GVP compatibility.
        If mismatched, it replaces with zero vectors of the correct shape.
        """
        expected_vector_dim = expected_edge_dims[1]  # e.g., 1 from (32, 1)

        # Case 1: edge_v missing or wrong shape
        if edge_v.dim() != 3 or edge_v.size(1) != expected_vector_dim:
            edge_v = torch.zeros(edge_s.size(0), expected_vector_dim, 3, device=edge_s.device)

        return edge_s, edge_v

    def _process_ligand_nodes(self, ligand_batch):
        """Process ligand nodes with GVP encoder"""
        lx_s = ligand_batch.x
        le_s = ligand_batch.edge_attr
        le_idx = ligand_batch.edge_index
        lx_v_dummy = torch.zeros(lx_s.size(0), 0, 3, device=lx_s.device)
        le_v_dummy = torch.zeros(le_s.size(0), 0, 3, device=le_s.device)
        lig_s, _ = self.ligand_encoder(lx_s, lx_v_dummy, le_idx, le_s, le_v_dummy)
        return lig_s

    def _cross_attention_interaction(self, p, l):
        """Cross-attention between protein and ligand"""
        embed_dim = self.hparams.protein_hidden_dims[0]
        
        if p.size(0) > 0 and l.size(0) > 0:
            # Both protein and ligand present
            p_batch = p.unsqueeze(0)  # [1, N_p, D]
            l_batch = l.unsqueeze(0)  # [1, N_l, D]
            
            # Bidirectional cross-attention
            # Ligand attends to protein
            l_enhanced, _ = self.protein_ligand_attn(l_batch, p_batch, p_batch)  # [1, N_l, D]
            # Protein attends to ligand
            p_enhanced, _ = self.ligand_protein_attn(p_batch, l_batch, l_batch)  # [1, N_p, D]
            
            # Pool representations
            l_pooled = l_enhanced.mean(dim=1)  # [1, D]
            p_pooled = p_enhanced.mean(dim=1)  # [1, D]
            
            # Fuse protein and ligand representations
            pl_combined = torch.cat([p_pooled, l_pooled], dim=-1)  # [1, 2*D]
            final_repr = self.pl_fusion(pl_combined)  # [1, D]
            
        elif p.size(0) > 0:
            # Only protein present
            p_pooled = p.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, D]
            # Use zero ligand representation
            l_zero = torch.zeros(1, embed_dim, device=self.device).unsqueeze(0)  # [1, 1, D]
            pl_combined = torch.cat([p_pooled.squeeze(1), l_zero.squeeze(1)], dim=-1)  # [1, 2*D]
            final_repr = self.pl_fusion(pl_combined)  # [1, D]
            
        elif l.size(0) > 0:
            # Only ligand present
            l_pooled = l.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, D]
            # Use zero protein representation
            p_zero = torch.zeros(1, embed_dim, device=self.device).unsqueeze(0)  # [1, 1, D]
            pl_combined = torch.cat([p_zero.squeeze(1), l_pooled.squeeze(1)], dim=-1)  # [1, 2*D]
            final_repr = self.pl_fusion(pl_combined)  # [1, D]
            
        else:
            # Neither protein nor ligand present
            final_repr = torch.zeros(1, embed_dim, device=self.device)  # [1, D]

        return final_repr.squeeze(0)  # [D]

    def _concatenation_interaction(self, p, l):
        """Simple concatenation-based interaction"""
        embed_dim = self.hparams.protein_hidden_dims[0]
        
        # Pool protein and ligand representations
        if p.size(0) > 0:
            p_pooled = p.mean(dim=0)  # [D]
        else:
            p_pooled = torch.zeros(embed_dim, device=self.device)  # [D]
            
        if l.size(0) > 0:
            l_pooled = l.mean(dim=0)  # [D]
        else:
            l_pooled = torch.zeros(embed_dim, device=self.device)  # [D]
        
        # Concatenate and fuse
        pl_combined = torch.cat([p_pooled, l_pooled], dim=-1)  # [2*D]
        final_repr = self.pl_fusion(pl_combined)  # [D]
        
        return final_repr
    
    def _log_predictions(self, batch, y_pred, y, stage):
        for i, (pred, true) in enumerate(zip(y_pred.tolist(), y.tolist())):
            name = batch['name'][i] if 'name' in batch else f"{stage}_sample_{i}"
            self.print(f"[{stage}] {name}: True Aff = {true:.3f}, Predicted = {pred:.3f}")
