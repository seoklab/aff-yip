import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class EGNNConv(MessagePassing):
    r"""Equivariant Graph Convolutional Layer from `E(n) Equivariant Graph
    Neural Networks <https://arxiv.org/abs/2102.09844>`__

    .. math::

        m_{ij}=\phi_e(h_i^l, h_j^l, ||x_i^l-x_j^l||^2, a_{ij})

        x_i^{l+1} = x_i^l + C\sum_{j\in\mathcal{N}(i)}(x_i^l-x_j^l)\phi_x(m_{ij})

        m_i = \sum_{j\in\mathcal{N}(i)} m_{ij}

        h_i^{l+1} = \phi_h(h_i^l, m_i)

    where :math:`h_i`, :math:`x_i`, :math:`a_{ij}` are node features, coordinate
    features, and edge features respectively. :math:`\phi_e`, :math:`\phi_h`, and
    :math:`\phi_x` are two-layer MLPs. :math:`C` is a constant for normalization,
    computed as :math:`1/|\mathcal{N}(i)|`.

    Parameters
    ----------
    in_size : int
        Input feature size; i.e. the size of :math:`h_i^l`.
    hidden_size : int
        Hidden feature size; i.e. the size of hidden layer in the two-layer MLPs in
        :math:`\phi_e, \phi_x, \phi_h`.
    out_size : int
        Output feature size; i.e. the size of :math:`h_i^{l+1}`.
    edge_feat_size : int, optional
        Edge feature size; i.e. the size of :math:`a_{ij}`. Default: 0.

    """

    def __init__(self, in_size, hidden_size, out_size, dropout=0.0, edge_feat_size=0):
        super(EGNNConv, self).__init__(aggr='add')

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the dist feature: ||x_i - x_j||
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, out_size),
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
        )

    def message(self, x_i, x_j, h_i, h_j, edge_attr=None):
        """message function for EGNN"""
        # compute distance and normalized difference
        x_diff = x_i - x_j
        dist = torch.norm(x_diff, dim=-1, keepdim=True)
        x_diff_norm = x_diff / (dist + 1e-7)
        
        # concat features for edge mlp
        if self.edge_feat_size > 0 and edge_attr is not None:
            f = torch.cat([h_i, h_j, dist, edge_attr], dim=-1)
        else:
            f = torch.cat([h_i, h_j, dist], dim=-1)

        msg_h = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h) * x_diff_norm

        return msg_h  # Only return node features for aggregation

    def forward(self, h, x, edge_index, edge_attr=None):
        r"""
        Description
        -----------
        Compute EGNN layer.

        Parameters
        ----------
        h : torch.Tensor
            The input node features of shape :math:`(N, h_n)`. :math:`N` is the number of
            nodes, and :math:`h_n` must be the same as in_size.
        x : torch.Tensor
            The coordinate features of shape :math:`(N, h_x)`. :math:`N` is the
            number of nodes, and :math:`h_x` can be any positive integer.
        edge_index : torch.Tensor
            Graph connectivity in COO format with shape [2, num_edges].
        edge_attr : torch.Tensor, optional
            Edge features of shape :math:`(M, h_e)`. :math:`M` is the number of
            edges, and :math:`h_e` must be the same as edge_feat_size.

        Returns
        -------
        h_out : torch.Tensor
            The output node features of shape :math:`(N, h_n')` where :math:`h_n'`
            is the same as out_size.
        x_out: torch.Tensor
            The output coordinate features of shape :math:`(N, h_x)` where :math:`h_x`
            is the same as the input coordinate feature dimension.
        """
        if self.edge_feat_size > 0:
            assert edge_attr is not None, "Edge features must be provided."
        
        # Propagate messages for node features
        msg_h = self.propagate(
            edge_index, x=x, h=h, edge_attr=edge_attr, size=None
        )
        
        # Update node features
        h_out = self.node_mlp(torch.cat([h, msg_h], dim=-1))
        
        # Update coordinates separately (mean aggregation for coordinates)
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        h_i, h_j = h[row], h[col]
        x_diff = x_i - x_j
        dist = torch.norm(x_diff, dim=-1, keepdim=True)
        x_diff_norm = x_diff / (dist + 1e-7)
        
        if self.edge_feat_size > 0 and edge_attr is not None:
            f = torch.cat([h_i, h_j, dist, edge_attr], dim=-1)
        else:
            f = torch.cat([h_i, h_j, dist], dim=-1)
        
        msg_h_coord = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h_coord) * x_diff_norm
        
        # Mean aggregation for coordinates
        from torch_scatter import scatter
        x_out = x + scatter(msg_x, row, dim=0, dim_size=x.size(0), reduce='mean')
        
        return h_out, x_out


class CoordEGNNConv(MessagePassing):
    def __init__(self, in_size, hidden_size, dropout=0.0, edge_feat_size=0):
        super(CoordEGNNConv, self).__init__(aggr='mean')

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the dist feature: ||x_i - x_j||
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
        )

    def message(self, x_i, x_j, h_i, h_j, edge_attr=None):
        # compute distance and normalized difference
        x_diff = x_i - x_j
        dist = torch.norm(x_diff, dim=-1, keepdim=True)
        x_diff_norm = x_diff / (dist + 1e-7)
        
        # concat features for edge mlp
        if self.edge_feat_size > 0 and edge_attr is not None:
            f = torch.cat([h_i, h_j, dist, edge_attr], dim=-1)
        else:
            f = torch.cat([h_i, h_j, dist], dim=-1)

        msg_h = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h) * x_diff_norm

        return msg_x

    def forward(self, h, x, edge_index, edge_attr=None):
        if self.edge_feat_size > 0:
            assert edge_attr is not None, "Edge features must be provided."
        
        # Propagate messages for coordinates
        coord_update = self.propagate(
            edge_index, x=x, h=h, edge_attr=edge_attr, size=None
        )
        
        return coord_update


class CoordEGNNConv_v2(MessagePassing):
    def __init__(self, in_size, hidden_size, dropout=0.0, edge_feat_size=0):
        super(CoordEGNNConv_v2, self).__init__(aggr='add')  # Changed to sum

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the dist feature: ||x_i - x_j||
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
        )

    def message(self, x_i, x_j, h_i, h_j, edge_attr=None):
        # compute distance and normalized difference
        x_diff = x_i - x_j
        dist = torch.norm(x_diff, dim=-1, keepdim=True)
        x_diff_norm = x_diff / (dist + 1e-7)
        
        # concat features for edge mlp
        if self.edge_feat_size > 0 and edge_attr is not None:
            f = torch.cat([h_i, h_j, dist, edge_attr], dim=-1)
        else:
            f = torch.cat([h_i, h_j, dist], dim=-1)

        msg_h = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h) * x_diff_norm

        return msg_x

    def forward(self, h, x, edge_index, edge_attr=None):
        if self.edge_feat_size > 0:
            assert edge_attr is not None, "Edge features must be provided."
        
        # Propagate messages for coordinates (sum aggregation)
        coord_update = self.propagate(
            edge_index, x=x, h=h, edge_attr=edge_attr, size=None
        )
        
        return coord_update


class CoordEGNNConv_v3(MessagePassing):
    def __init__(self, in_size, hidden_size, dropout=0.0, edge_feat_size=0):
        super(CoordEGNNConv_v3, self).__init__(aggr='add')

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the dist feature: ||x_i - x_j||
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
        )
        
        # query
        self.linear_q = nn.Linear(in_size, hidden_size, bias=False)
        
        # key
        self.linear_k = nn.Linear(in_size + edge_feat_size + 1, hidden_size, bias=False)

    def message(self, x_i, x_j, h_i, h_j, edge_index, edge_attr=None):
        # compute distance and normalized difference
        x_diff = x_i - x_j
        dist = torch.norm(x_diff, dim=-1, keepdim=True)
        x_diff_norm = x_diff / (dist + 1e-7)
        
        # concat features for edge mlp
        if self.edge_feat_size > 0 and edge_attr is not None:
            f = torch.cat([h_i, h_j, dist, edge_attr], dim=-1)
            k_raw = torch.cat([h_i, dist, edge_attr], dim=-1)
        else:
            f = torch.cat([h_i, h_j, dist], dim=-1)
            k_raw = torch.cat([h_i, dist], dim=-1)

        msg_h = self.edge_mlp(f)
        
        msg_k = self.linear_k(k_raw)
        msg_q = self.linear_q(h_j)
        
        # Compute attention scores
        msg_e = (msg_k * msg_q).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        
        # Apply softmax attention
        row, col = edge_index
        att = softmax(msg_e.squeeze(-1), row, num_nodes=h_j.size(0))
        
        msg_v = self.coord_mlp(msg_h) * x_diff_norm
        
        # Apply attention weights
        msg_x = att.unsqueeze(-1) * msg_v

        return msg_x

    def forward(self, h, x, edge_index, edge_attr=None):
        if self.edge_feat_size > 0:
            assert edge_attr is not None, "Edge features must be provided."
        
        # Propagate messages for coordinates with attention
        coord_update = self.propagate(
            edge_index, x=x, h=h, edge_index=edge_index, edge_attr=edge_attr, size=None
        )
        
        return coord_update