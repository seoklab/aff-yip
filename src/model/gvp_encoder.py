# src/model/gvp_encoder.py
import torch
import torch.nn as nn
from src.gvp import GVPConvLayer, LayerNorm, Dropout

class GVPGraphEncoder(nn.Module):
    def __init__(self, 
                 node_dims=(6, 3),         # (scalar, vector)
                 edge_dims=(41, 3),        # (scalar, vector)
                 hidden_dims=(128, 16),    # (scalar, vector) hidden dims
                 num_layers=3,
                 drop_rate=0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.LayerNorm(node_dims[0]),
            nn.Linear(node_dims[0], hidden_dims[0])
        )
        self.layers = nn.ModuleList([
            GVPConvLayer(hidden_dims, edge_dims, 
                         drop_rate=drop_rate)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(hidden_dims)
        self.dropout = Dropout(drop_rate)

    def forward(self, x_s, x_v, edge_index, edge_s, edge_v):
        # Project scalar node features to hidden dim
        x_s = self.input_proj(x_s)

        x = (x_s, x_v)
        edge = (edge_s, edge_v)

        for layer in self.layers:
            x = layer(x, edge_index, edge)

        x = self.norm(self.dropout(x))
        return x  # (scalar, vector) tuple