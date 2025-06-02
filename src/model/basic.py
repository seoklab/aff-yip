import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from src.model.gvp_encoder import GVPGraphEncoder

class AFFModel_GVP(pl.LightningModule):
    def __init__(self,
                 protein_node_dims=(6, 3),
                 protein_edge_dims=(32, 3),
                 ligand_node_dims=(46, 0),
                 ligand_edge_dims=(9, 0),
                 protein_hidden_dims=(128, 16),
                 ligand_hidden_dims=(128, 0),
                 num_gvp_layers=3,
                 dropout=0.1,
                 lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

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

        self.attn = nn.MultiheadAttention(
            embed_dim=protein_hidden_dims[0],
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
        protein_graphs = [item['protein_water'] for item in batch]
        ligand_graphs = [item['ligand'] for item in batch]
        affinities = torch.tensor([item['affinity'] for item in batch], dtype=torch.float32, device=self.device)

        protein_batch = Batch.from_data_list(protein_graphs).to(self.device)
        ligand_batch = Batch.from_data_list(ligand_graphs).to(self.device)

        # === Protein encoding ===
        x_s, x_v = protein_batch.node_s, protein_batch.node_v
        e_s, e_v = protein_batch.edge_s, protein_batch.edge_v
        edge_index = protein_batch.edge_index
        prot_s, _ = self.protein_encoder(x_s, x_v, edge_index, e_s, e_v)

        # === Ligand encoding ===
        lx_s = ligand_batch.x  # shape [N, 46]
        le_s = ligand_batch.edge_attr  # shape [E, 9]
        le_idx = ligand_batch.edge_index
        lig_s, _ = self.ligand_encoder(lx_s, torch.zeros_like(lx_s[..., :0]), le_idx, le_s, torch.zeros_like(le_s[..., :0]))

        # === Group nodes by batch
        prot_s_list = torch.split(prot_s, torch.bincount(protein_batch.batch).tolist())
        lig_s_list = torch.split(lig_s, torch.bincount(ligand_batch.batch).tolist())

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
            name = batch[i].get("name", f"{stage}_sample_{i}")
            self.print(f"[{stage}] {name}: True Aff = {true:.3f}, Predicted = {pred:.3f}")

    def training_step(self, batch, batch_idx):
        y_pred, y = self(batch)
        loss = self.loss_fn(y_pred, y)
        self._log_predictions(batch, y_pred, y, "Train")
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, y = self(batch)
        val_loss = self.loss_fn(y_pred, y)
        self._log_predictions(batch, y_pred, y, "Val")
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        y_pred, y = self(batch)
        test_loss = self.loss_fn(y_pred, y)
        self._log_predictions(batch, y_pred, y, "Test")
        self.log("test_loss", test_loss, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)