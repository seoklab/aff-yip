"""
Portions of this code were adapted from 
https://github.com/jingraham/neurips19-graph-protein-design

PyTorch Lightning implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm

from gvp import * 


def gather_edges(edges, neighbor_idx):
    """
    Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    """
    # Flatten neighbors to [B, N*K]
    neighbors_flat = neighbor_idx.view(neighbor_idx.shape[0], -1)
    
    # Create batch indices [B, N*K]
    batch_size = edges.shape[0]
    batch_indices = torch.arange(batch_size, device=edges.device).view(-1, 1)
    batch_indices = batch_indices.repeat(1, neighbor_idx.shape[1] * neighbor_idx.shape[2])
    
    # Gather from [B, N, N, C] to [B, N*K, C] using batch_indices and neighbors_flat
    neighbors_flat_edges = edges[batch_indices.view(-1), 
                                 neighbors_flat.view(-1) // edges.shape[2], 
                                 neighbors_flat.view(-1) % edges.shape[2]]
    
    # Reshape to [B, N, K, C]
    edge_features = neighbors_flat_edges.view(batch_size, 
                                              neighbor_idx.shape[1], 
                                              neighbor_idx.shape[2], 
                                              -1)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    """
    Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    """
    # Flatten and expand indices per batch [B,N,K] => [B,NK]
    neighbors_flat = neighbor_idx.view(neighbor_idx.shape[0], -1)
    
    # Create batch indices [B, NK]
    batch_size = nodes.shape[0]
    batch_indices = torch.arange(batch_size, device=nodes.device).view(-1, 1)
    batch_indices = batch_indices.repeat(1, neighbors_flat.shape[1])
    
    # Gather and re-pack
    neighbor_features = nodes[batch_indices.view(-1), neighbors_flat.view(-1)]
    neighbor_features = neighbor_features.view(batch_size, 
                                              neighbor_idx.shape[1], 
                                              neighbor_idx.shape[2], 
                                              -1)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx, nv_nodes, nv_neighbors):
    """
    Concatenate node features with neighbor features
    """
    h_nodes = gather_nodes(h_nodes, E_idx)
    return vs_concat(h_neighbors, h_nodes, nv_neighbors, nv_nodes)


def autoregressive_mask(E_idx):
    """
    Generates a mask for autoregressive models where node i can only attend
    to nodes j < i in the sequence
    """
    N_nodes = E_idx.shape[1]
    ii = torch.arange(N_nodes, device=E_idx.device)
    ii = ii.view(1, -1, 1)
    mask = E_idx - ii < 0
    mask = mask.float()
    return mask


def normalize(tensor, axis=-1):
    """
    Normalizes a tensor along a dimension
    """
    return F.normalize(tensor, p=2, dim=axis)


class PositionalEncodings(nn.Module):
    """
    Positional encodings for node positions in the graph
    """
    def __init__(self, num_embeddings, period_range=[2, 1000]):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range

    def forward(self, E_idx):
        # Get indices i-j
        N_batch = E_idx.shape[0]
        N_nodes = E_idx.shape[1]
        N_neighbors = E_idx.shape[2]
        
        ii = torch.arange(N_nodes, device=E_idx.device).float().view(1, -1, 1)
        d = (E_idx.float() - ii).unsqueeze(-1)
        
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, device=E_idx.device).float()
            * -(np.log(10000.0) / self.num_embeddings)
        )
        angles = d * frequency.view(1, 1, 1, -1)
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E


class StructuralFeatures(nn.Module):
    """
    Extracts structural features from protein backbone coordinates
    """
    def __init__(self, node_features, edge_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=30):
        super(StructuralFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        
        # Normalization and embedding
        vo, so = node_features
        ve, se = edge_features
        self.node_embedding = GVP(vi=3, vo=vo, so=so, nlv=None, nls=None)
        self.edge_embedding = GVP(vi=1, vo=ve, so=se, nlv=None, nls=None)
        self.norm_nodes = nn.LayerNorm(so)
        self.norm_edges = nn.LayerNorm(se)
    
    def _dist(self, X, mask, eps=1E-6):
        """
        Pairwise euclidean distances
        """
        mask = mask.float()
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(-D_adjust, k=min(self.top_k, X.shape[1]))
        D_neighbors = -D_neighbors
        
        # Create mask for neighbor positions
        # Create batch indices for selecting edges
        batch_indices = torch.arange(D.shape[0], device=D.device).view(-1, 1, 1)
        batch_indices = batch_indices.repeat(1, D.shape[1], min(self.top_k, X.shape[1]))
        
        # Create node indices for selecting edges
        node_indices = torch.arange(D.shape[1], device=D.device).view(1, -1, 1)
        node_indices = node_indices.repeat(D.shape[0], 1, min(self.top_k, X.shape[1]))
        
        # Gather edge masks
        mask_gather = mask_2D[batch_indices, node_indices, E_idx]
        mask_neighbors = mask_gather.unsqueeze(-1)
        
        return D_neighbors, E_idx, mask_neighbors
   
    def _directions(self, X, E_idx):
        """
        Compute directions between nodes
        """
        X_neighbors = gather_nodes(X, E_idx)
        dX = X_neighbors - X.unsqueeze(2)
        return normalize(dX, axis=-1)
        
    def _rbf(self, D):
        """
        Distance radial basis function
        """
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view(1, 1, 1, -1)
        D_sigma = (D_max - D_min) / D_count
        D_expand = D.unsqueeze(-1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        
        return RBF
    
    def _orientations(self, X):
        """
        Compute orientation features
        """
        forward = normalize(X[:, 1:] - X[:, :-1])
        backward = normalize(X[:, :-1] - X[:, 1:])
        forward = F.pad(forward, (0, 0, 0, 1))
        backward = F.pad(backward, (0, 0, 1, 0))
        return torch.stack([forward, backward], dim=-1)  # B, N, 3, 2
        
    def _sidechains(self, X):
        """
        Compute sidechain features
        """
        n, origin, c = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :]
        c, n = normalize(c - origin), normalize(n - origin)
        bisector = normalize(c + n)
        perp = normalize(torch.cross(c, n))
        vec = -bisector * torch.sqrt(torch.tensor(1/3, device=X.device)) - perp * torch.sqrt(torch.tensor(2/3, device=X.device))
        return vec  # B, N, 3

    def _dihedrals(self, X, eps=1e-7):
        """
        Compute dihedral angles
        """
        # First 3 coordinates are N, CA, C
        X = X[:, :, :3, :].reshape(X.shape[0], 3*X.shape[1], 3)
        
        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = normalize(dX, axis=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        
        # Backbone normals
        n_2 = normalize(torch.cross(u_2, u_1), axis=-1)
        n_1 = normalize(torch.cross(u_1, u_0), axis=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1, 2))
        D = D.reshape(D.shape[0], int(D.shape[1]/3), 3)
        
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 2)
        return D_features
        
    def forward(self, X, mask):
        """
        Featurize coordinates as an attributed graph
        """
        # Build k-Nearest Neighbors graph
        X_ca = X[:, :, 1, :]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask)
        
        # Pairwise features
        E_directions = self._directions(X_ca, E_idx)
        RBF = self._rbf(D_neighbors)
        E_positional = self.embeddings(E_idx)
        
        # Full backbone angles
        V_dihedrals = self._dihedrals(X)
        V_orientations = self._orientations(X_ca)
        V_sidechains = self._sidechains(X)
        
        # Combine features
        V_orientations_combined = torch.cat([V_sidechains.unsqueeze(-1), V_orientations], dim=-1)
        V = merge(V_orientations_combined, V_dihedrals)
        E = torch.cat([E_directions, RBF, E_positional], dim=-1)
        
        # Embed the nodes and edges
        Vv, Vs = self.node_embedding(V, return_split=True)
        V = merge(Vv, self.norm_nodes(Vs))
        
        Ev, Es = self.edge_embedding(E, return_split=True)
        E = merge(Ev, self.norm_edges(Es))
        
        return V, E, E_idx


class MPNNLayer(nn.Module):
    """
    Message Passing Neural Network Layer for GVP features
    """
    def __init__(self, vec_in, num_hidden, dropout=0.1):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.vec_in = vec_in
        self.vo, self.so = vo, so = num_hidden
        
        self.norm = nn.ModuleList([GVPLayerNorm(vo) for _ in range(2)])
        self.dropout = GVPDropout(dropout, vo)
        
        # W_EV: Takes messages with vector channel vec_in plus receiver node with vec channel vo
        self.W_EV = nn.Sequential(
            GVP(vi=vec_in+vo, vo=vo, so=so), 
            GVP(vi=vo, vo=vo, so=so),
            GVP(vi=vo, vo=vo, so=so, nls=None, nlv=None)
        )
        
        # W_dh: Node update function
        self.W_dh = nn.Sequential(
            GVP(vi=vo, vo=2*vo, so=4*so),
            GVP(vi=2*vo, vo=vo, so=so, nls=None, nlv=None)
        )
        
    def forward(self, h_V, h_M, mask_V=None, mask_attend=None, train=True):
        """
        :param h_V: Node features
        :param h_M: Messages from neighbors
        :param mask_V: Mask for nodes
        :param mask_attend: Mask for attention
        :param train: Whether in training mode
        """
        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_M.shape[-2], -1)
        h_EV = vs_concat(h_V_expand, h_M, self.vo, self.vec_in)
        h_message = self.W_EV(h_EV)
        
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1).float() * h_message
        
        # Aggregate messages
        dh = torch.mean(h_message, dim=2)
        
        # Update node representations
        h_V = self.norm[0](h_V + self.dropout(dh) if train else dh)

        # Position-wise feedforward
        dh = self.W_dh(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh) if train else dh)

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1).float()
            h_V = mask_V * h_V
            
        return h_V


class Encoder(nn.Module):
    """
    Graph encoder module for protein structure
    """
    def __init__(self, node_features, edge_features, num_layers=3, dropout=0.1):
        super(Encoder, self).__init__()
        
        # Hyperparameters
        self.nv, ns = node_features
        self.ev, _ = edge_features
        
        # Encoder layers 
        self.vglayers = nn.ModuleList([                               
            MPNNLayer(self.nv + self.ev, node_features, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, h_V, h_E, E_idx, mask, train=False):
        """
        :param h_V: Node features
        :param h_E: Edge features
        :param E_idx: Edge indices
        :param mask: Node mask
        :param train: Whether in training mode
        """        
        # Create attention mask
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        
        # Apply graph message passing layers
        for layer in self.vglayers:
            h_M = cat_neighbors_nodes(h_V, h_E, E_idx, self.nv, self.ev)
            h_V = layer(h_V, h_M, mask_V=mask, mask_attend=mask_attend, train=train)
            
        return h_V


class Decoder(nn.Module):
    """
    Autoregressive decoder module for protein sequence generation
    This module takes structural features and generates amino acid sequences
    """
    def __init__(self, node_features, edge_features, s_features, 
                 num_layers=3, dropout=0.1):
        super(Decoder, self).__init__()
        
        # Hyperparameters
        self.nv, self.ns = node_features
        self.ev, self.es = edge_features
        self.sv, self.ss = s_features
        
        # Decoder layers
        self.vglayers = nn.ModuleList([
            MPNNLayer(self.nv + self.ev, node_features, dropout=dropout)
            for _ in range(num_layers)              
        ])
        
    def forward(self, h_V, h_S, h_E, E_idx, mask, train=False):
        """
        :param h_V: Node features from encoder
        :param h_S: Sequence features (input for autoregressive decoder)
        :param h_E: Edge features
        :param E_idx: Edge indices
        :param mask: Node mask
        :param train: Whether in training mode
        """
        # Concatenate sequence embeddings for autoregressive decoder
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx, 0, self.ev)

        # Build encoder embeddings
        zeros_like_S = torch.zeros_like(h_S)
        h_ES_encoder = cat_neighbors_nodes(zeros_like_S, h_E, E_idx, self.sv, self.ev)
        h_ESV_encoder = cat_neighbors_nodes(h_V, h_ES_encoder, E_idx, self.nv, self.sv + self.ev)

        # Decoder uses masked self-attention
        mask_attend = autoregressive_mask(E_idx).unsqueeze(-1)
        mask_1D = mask.unsqueeze(-1).unsqueeze(-1).float()
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        h_ESV_encoder_fw = mask_fw * h_ESV_encoder
        
        for layer in self.vglayers:
            # Masked positions attend to encoder information, unmasked see. 
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx, self.nv, self.ev)
            h_M = mask_bw * h_ESV + h_ESV_encoder_fw
            h_V = layer(h_V, h_M, mask_V=mask, train=train)
        
        return h_V
        
    def sample(self, h_V, h_E, E_idx, mask, W_s, W_out, temperature=0.1):
        """
        Autoregressive sampling of sequences
        """
        device = h_V.device
        
        # Setup masks for autoregressive generation
        mask_attend = autoregressive_mask(E_idx).unsqueeze(-1)
        mask_1D = mask.unsqueeze(-1).unsqueeze(-1).float()
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        
        N_batch, N_nodes = h_V.shape[0], h_V.shape[1] 
        
        # Initialize sequence and sequence embeddings
        h_S = torch.zeros((N_batch, N_nodes, self.ss), device=device, dtype=torch.float32)
        S = torch.zeros((N_batch, N_nodes), device=device, dtype=torch.long)
        
        # Initialize a list of vectors for storing hidden activations 
        h_V_stack = [torch.split(h_V, 1, dim=1)] + [
            torch.split(torch.zeros_like(h_V), 1, dim=1) 
            for _ in range(len(self.vglayers))
        ]
        
        # Autoregressive sampling loop
        for t in tqdm(range(N_nodes)):
            # Extract hidden states for current step
            E_idx_t = E_idx[:, t:t+1, :]
            h_E_t = h_E[:, t:t+1, :, :]
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t, 0, self.ev)
            
            # Stale relational features for future states
            h_ESV_encoder_t = mask_fw[:, t:t+1, :, :] * cat_neighbors_nodes(h_V, h_ES_t, E_idx_t, self.nv, self.ev)
            
            # Process through encoder layers
            for l, layer in enumerate(self.vglayers):
                # Updated relational features for future states
                h_V_stack_tensor = torch.cat([v for v in h_V_stack[l]], dim=1)
                h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack_tensor, h_ES_t, E_idx_t, self.nv, self.ev)
                h_V_t = h_V_stack[l][t]
                h_ESV_t = mask_bw[:, t:t+1, :, :] * h_ESV_decoder_t + h_ESV_encoder_t
                mask_to_pass = mask[:, t:t+1]
                h_V_stack[l+1][t] = layer(h_V_t, h_ESV_t, mask_V=mask_to_pass, train=False)
            
            # Sampling step
            h_V_t = h_V_stack[-1][t].squeeze(1)
            logits = W_out(h_V_t) / temperature
            probs = F.softmax(logits, dim=-1)
            S_t = torch.multinomial(probs, 1).squeeze(-1)

            # Update
            h_S[:, t, :] = W_s(S_t)
            S[:, t] = S_t
        
        return S


class MQAModel(pl.LightningModule):
    """
    Model Quality Assessment model
    Predicts the quality of a protein structure
    """
    def __init__(self, node_features, edge_features,
                 hidden_dim, num_layers=3, k_neighbors=30, dropout=0.1,
                 learning_rate=1e-3):
        super(MQAModel, self).__init__()
        
        # Hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.nv, self.ns = node_features
        self.hv, self.hs = hidden_dim
        self.ev, self.es = edge_features
        
        # Featurization layers
        self.features = StructuralFeatures(node_features, edge_features, top_k=k_neighbors)
    
        # Embedding layers
        self.W_s = nn.Embedding(20, self.hs)        
        self.W_v = GVP(vi=self.nv, vo=self.hv, so=self.hs, nls=None, nlv=None)
        self.W_e = GVP(vi=self.ev, vo=self.ev, so=self.hs, nls=None, nlv=None)
        
        # Encoder
        self.encoder = Encoder(hidden_dim, edge_features, num_layers=num_layers, dropout=dropout)
        
        # Output layers
        self.W_V_out = GVP(vi=self.hv, vo=0, so=self.hs, nls=None, nlv=None)
        
        self.dense = nn.Sequential(
            nn.Linear(self.hs, 2 * self.hs),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.hs, 2 * self.hs),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(2 * self.hs),
            nn.Linear(2 * self.hs, 1)
        )
    
    def forward(self, X, S, mask, train=False):
        """
        Forward pass of the MQA model
        :param X: Backbone coordinates [B, N, 4, 3]
        :param S: Sequence indices [B, N]
        :param mask: Mask for nodes [B, N]
        :param train: Whether in training mode
        :return: Quality score [B]
        """
        # Get structural features
        V, E, E_idx = self.features(X, mask)
        
        # Embed sequence
        h_S = self.W_s(S)
        
        # Concatenate sequence and structure features
        V = vs_concat(V, h_S, self.nv, 0)
        
        # Apply embeddings
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        
        # Encode the graph
        h_V = self.encoder(h_V, h_E, E_idx, mask, train=train)
        
        # Get embeddings for output
        h_V_out = self.W_V_out(h_V)
        mask = mask.unsqueeze(-1)  # [B, N, 1]

        # Pool node features based on mode
        if train:
            h_V_out = torch.mean(h_V_out * mask, dim=1)  # [B, N, D] -> [B, D]
        else:
            # Safe mean that handles zero masks
            h_V_out = torch.sum(h_V_out * mask, dim=1)  # [B, N, D] -> [B, D]
            mask_sum = torch.sum(mask, dim=1)  # [B, 1]
            mask_sum = torch.clamp(mask_sum, min=1.0)  # Avoid division by zero
            h_V_out = h_V_out / mask_sum  # [B, D]
        
        # Final output
        out = self.dense(h_V_out).squeeze(-1) + 0.5  # [B]
        
        return out
    
    def training_step(self, batch, batch_idx):
        """
        Training step for the model
        """
        X, S, mask, y = batch
        pred = self(X, S, mask, train=True)
        loss = F.mse_loss(pred, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model
        """
        X, S, mask, y = batch
        pred = self(X, S, mask, train=False)
        loss = F.mse_loss(pred, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step for the model
        """
        X, S, mask, y = batch
        pred = self(X, S, mask, train=False)
        loss = F.mse_loss(pred, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizers for the model
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


class CPDModel(pl.LightningModule):
    """
    Conditional Protein Design model
    Generates protein sequences given a backbone structure
    """
    def __init__(self, node_features, edge_features,
                 hidden_dim, num_layers=3, num_letters=20, k_neighbors=30, 
                 dropout=0.1, learning_rate=1e-3):
        super(CPDModel, self).__init__()
        
        # Hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.nv, self.ns = node_features
        self.hv, self.hs = hidden_dim
        self.ev, self.es = edge_features
        self.num_letters = num_letters
        
        # Featurization layers
        self.features = StructuralFeatures(node_features, edge_features, top_k=k_neighbors)
    
        # Embedding layers
        self.W_v = GVP(vi=self.nv, vo=self.hv, so=self.hs, nls=None, nlv=None)
        self.W_e = GVP(vi=self.ev, vo=self.ev, so=self.hs, nls=None, nlv=None)
        self.W_s = nn.Embedding(num_letters, self.hs)
        
        # Encoder and decoder
        self.encoder = Encoder(hidden_dim, edge_features, num_layers=num_layers)
        self.decoder = Decoder(hidden_dim, edge_features, s_features=(0, self.hs), num_layers=num_layers)
        
        # Output layer
        self.W_out = GVP(vi=self.hv, vo=0, so=num_letters, nls=None, nlv=None)


    def forward(self, X, S, mask, train=False):
        """
        Forward pass of the CPD model
        :param X: Backbone coordinates [B, N, 4, 3]
        :param S: Sequence indices [B, N]
        :param mask: Mask for nodes [B, N]
        :param train: Whether in training mode
        :return: Logits for amino acid prediction [B, N, num_letters]
        """
        # Get structural features
        V, E, E_idx = self.features(X, mask)
        
        # Apply embeddings
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        
        # Encode the graph
        h_V = self.encoder(h_V, h_E, E_idx, mask, train=train)
        
        # Embed sequence for decoder
        h_S = self.W_s(S)
        
        # Decode to predict sequence
        h_V = self.decoder(h_V, h_S, h_E, E_idx, mask, train=train)
        
        # Get logits
        logits = self.W_out(h_V)
        
        return logits
    
    def sample(self, X, mask=None, temperature=0.1):
        """
        Sample sequences from the model
        :param X: Backbone coordinates [B, N, 4, 3]
        :param mask: Mask for nodes [B, N]
        :param temperature: Sampling temperature (lower = more deterministic)
        :return: Sampled sequences [B, N]
        """
        # Get structural features
        V, E, E_idx = self.features(X, mask)
        
        # Apply embeddings
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        
        # Encode the graph
        h_V = self.encoder(h_V, h_E, E_idx, mask, train=False)
        
        # Sample sequences
        return self.decoder.sample(h_V, h_E, E_idx, mask, 
                                   W_s=self.W_s, W_out=self.W_out, 
                                   temperature=temperature)
    
    def training_step(self, batch, batch_idx):
        """
        Training step for the model
        """
        X, S, mask = batch
        logits = self(X, S, mask, train=True)
        loss = F.cross_entropy(
            logits.reshape(-1, self.num_letters),
            S.reshape(-1),
            reduction='none'
        )
        # Apply mask
        mask_flat = mask.reshape(-1)
        loss = torch.sum(loss * mask_flat) / torch.sum(mask_flat)
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model
        """
        X, S, mask = batch
        logits = self(X, S, mask, train=False)
        loss = F.cross_entropy(
            logits.reshape(-1, self.num_letters),
            S.reshape(-1),
            reduction='none'
        )
        # Apply mask
        mask_flat = mask.reshape(-1)
        loss = torch.sum(loss * mask_flat) / torch.sum(mask_flat)
        
        # Calculate accuracy
        pred = logits.argmax(dim=-1)
        correct = (pred == S) * mask
        acc = torch.sum(correct) / torch.sum(mask)
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step for the model
        """
        X, S, mask = batch
        logits = self(X, S, mask, train=False)
        loss = F.cross_entropy(
            logits.reshape(-1, self.num_letters),
            S.reshape(-1),
            reduction='none'
        )
        # Apply mask
        mask_flat = mask.reshape(-1)
        loss = torch.sum(loss * mask_flat) / torch.sum(mask_flat)
        
        # Calculate accuracy
        pred = logits.argmax(dim=-1)
        correct = (pred == S) * mask
        acc = torch.sum(correct) / torch.sum(mask)
        
        # Calculate per-position accuracy
        per_pos_acc = torch.sum(correct, dim=0) / torch.sum(mask, dim=0)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizers for the model
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


# Aliases for compatibility with the original code
VGEncoder = Encoder
VGDecoder = Decoder