import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from src.model.gvp_encoder import GVPGraphEncoderHybrid as GVPGraphEncoder

class AFFModel_GVP(pl.LightningModule):
    def __init__(self,
                 protein_node_dims=(6, 3),      # matches your data: node_s=[N,6], node_v=[N,3,3]
                 protein_edge_dims=(32, 1),     # matches your data: edge_s=[E,32], edge_v=[E,1,3]
                 ligand_node_dims=(46, 0),      # ligand has no vector features
                 ligand_edge_dims=(9, 0),       # ligand edges have no vector features
                 protein_hidden_dims=(128, 16),
                 ligand_hidden_dims=(128, 0),   # ligand output has no vector features
                 num_gvp_layers=3,
                 dropout=0.1,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.protein_encoder = GVPGraphEncoder(
            node_dims=protein_node_dims,
            edge_dims=(32, 1),  # Fixed: match your data edge_v=[E,1,3]
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

        self.attn = nn.MultiheadAttention(
            embed_dim=protein_hidden_dims[0],  # 128
            num_heads=4,
            batch_first=True
        )

        self.regressor = nn.Sequential(
            nn.Linear(protein_hidden_dims[0], protein_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(protein_hidden_dims[0], 1)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        protein_batch = batch['protein_water'].to(self.device)
        ligand_batch = batch['ligand'].to(self.device)
        affinities = batch['affinity'].to(self.device)

        # === Protein encoding ===
        x_s, x_v = protein_batch.node_s, protein_batch.node_v  # [N,6], [N,3,3]
        e_s, e_v = protein_batch.edge_s, protein_batch.edge_v  # [E,32], [E,1,3]
        edge_index = protein_batch.edge_index  # [2,E]
        
        # Debug prints (remove after fixing)
        # print(f"Protein - x_s: {x_s.shape}, x_v: {x_v.shape}")
        # print(f"Protein - e_s: {e_s.shape}, e_v: {e_v.shape}")
        
        prot_s, prot_v = self.protein_encoder(x_s, x_v, edge_index, e_s, e_v)

        # === Ligand encoding ===
        lx_s = ligand_batch.x  # [N, 46]
        le_s = ligand_batch.edge_attr  # [E, 9]
        le_idx = ligand_batch.edge_index  # [2, E]
        
        # Create dummy vector features for ligand (since it has none)
        lx_v_dummy = torch.zeros(lx_s.size(0), 0, 3, device=lx_s.device)  # [N, 0, 3]
        le_v_dummy = torch.zeros(le_s.size(0), 0, 3, device=le_s.device)  # [E, 0, 3]
        
        # Debug prints
        # print(f"Ligand - lx_s: {lx_s.shape}, lx_v_dummy: {lx_v_dummy.shape}")
        # print(f"Ligand - le_s: {le_s.shape}, le_v_dummy: {le_v_dummy.shape}")
        
        lig_s, lig_v = self.ligand_encoder(lx_s, lx_v_dummy, le_idx, le_s, le_v_dummy)

        # === Group nodes by batch ===
        prot_batch_sizes = torch.bincount(protein_batch.batch).tolist()
        lig_batch_sizes = torch.bincount(ligand_batch.batch).tolist()
        
        prot_s_list = torch.split(prot_s, prot_batch_sizes)
        lig_s_list = torch.split(lig_s, lig_batch_sizes)

        preds = []
        for p, l in zip(prot_s_list, lig_s_list):
            # ligand as query, protein as key/value
            q = l.unsqueeze(0)  # [1, L, D]
            k = v = p.unsqueeze(0)  # [1, P, D]
            out, _ = self.attn(q, k, v)  # [1, L, D]
            pooled = out.mean(dim=1)  # [1, D]
            preds.append(self.regressor(pooled).squeeze())

        y_pred = torch.stack(preds)  # [B]
        return y_pred, affinities

    def _log_predictions(self, batch, y_pred, y, stage):
        for i, (pred, true) in enumerate(zip(y_pred.tolist(), y.tolist())):
            name = batch['name'][i] if 'name' in batch and i < len(batch['name']) else f"{stage}_sample_{i}"
            self.print(f"[{stage}] {name}: True Aff = {true:.3f}, Predicted = {pred:.3f}")

    def training_step(self, batch, batch_idx):
        y_pred, y = self(batch)
        loss = self.loss_fn(y_pred, y)
        actual_batch_size = len(batch['name']) if 'name' in batch else y.size(0)
        self._log_predictions(batch, y_pred, y, "Train")
        self.log("train_loss", loss, prog_bar=True, batch_size=actual_batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, y = self(batch)
        val_loss = self.loss_fn(y_pred, y)
        actual_batch_size = len(batch['name']) if 'name' in batch else y.size(0)
        self._log_predictions(batch, y_pred, y, "Val")
        self.log("val_loss", val_loss, prog_bar=True, batch_size=actual_batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        y_pred, y = self(batch)
        test_loss = self.loss_fn(y_pred, y)
        actual_batch_size = len(batch['name']) if 'name' in batch else y.size(0)
        self._log_predictions(batch, y_pred, y, "Test")
        self.log("test_loss", test_loss, prog_bar=True, batch_size=actual_batch_size)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# Alternative version with better error handling and debugging
class AFFModel_GVP_Debug(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Use the corrected dimensions
        protein_edge_dims = (32, 1)  # Your data shows edge_v=[E,1,3]
        
        self.protein_encoder = GVPGraphEncoder(
            node_dims=(6, 3),
            edge_dims=protein_edge_dims,
            hidden_dims=(128, 16),
            num_layers=3,
            drop_rate=0.1
        )

        self.ligand_encoder = GVPGraphEncoder(
            node_dims=(46, 0),
            edge_dims=(9, 0),
            hidden_dims=(128, 0),
            num_layers=3,
            drop_rate=0.1
        )

        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        try:
            protein_batch = batch['protein_water'].to(self.device)
            ligand_batch = batch['ligand'].to(self.device)
            
            # Protein encoding with dimension checks
            x_s, x_v = protein_batch.node_s, protein_batch.node_v
            e_s, e_v = protein_batch.edge_s, protein_batch.edge_v
            
            # Verify dimensions match expected
            assert x_s.dim() == 2 and x_s.size(1) == 6, f"Expected node_s [N,6], got {x_s.shape}"
            assert x_v.dim() == 3 and x_v.shape[1:] == (3, 3), f"Expected node_v [N,3,3], got {x_v.shape}"
            assert e_s.dim() == 2 and e_s.size(1) == 32, f"Expected edge_s [E,32], got {e_s.shape}"
            assert e_v.dim() == 3 and e_v.shape[1:] == (1, 3), f"Expected edge_v [E,1,3], got {e_v.shape}"
            
            prot_s, prot_v = self.protein_encoder(x_s, x_v, protein_batch.edge_index, e_s, e_v)
            
            # Ligand encoding
            lx_s = ligand_batch.x
            le_s = ligand_batch.edge_attr
            
            # Create proper dummy tensors
            lx_v_dummy = torch.zeros(lx_s.size(0), 0, 3, device=lx_s.device, dtype=lx_s.dtype)
            le_v_dummy = torch.zeros(le_s.size(0), 0, 3, device=le_s.device, dtype=le_s.dtype)
            
            lig_s, lig_v = self.ligand_encoder(lx_s, lx_v_dummy, ligand_batch.edge_index, le_s, le_v_dummy)
            
            # Rest of forward pass...
            prot_batch_sizes = torch.bincount(protein_batch.batch).tolist()
            lig_batch_sizes = torch.bincount(ligand_batch.batch).tolist()
            
            prot_s_list = torch.split(prot_s, prot_batch_sizes)
            lig_s_list = torch.split(lig_s, lig_batch_sizes)

            preds = []
            for p, l in zip(prot_s_list, lig_s_list):
                q = l.unsqueeze(0)
                k = v = p.unsqueeze(0)
                out, _ = self.attn(q, k, v)
                pooled = out.mean(dim=1)
                preds.append(self.regressor(pooled).squeeze())

            return torch.stack(preds), batch['affinity'].to(self.device)
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            print(f"Protein batch keys: {protein_batch.keys if hasattr(protein_batch, 'keys') else 'No keys'}")
            print(f"Ligand batch keys: {ligand_batch.keys if hasattr(ligand_batch, 'keys') else 'No keys'}")
            raise e

    def training_step(self, batch, batch_idx):
        y_pred, y = self(batch)
        return self.loss_fn(y_pred, y)

    def validation_step(self, batch, batch_idx):
        y_pred, y = self(batch)
        val_loss = self.loss_fn(y_pred, y)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)