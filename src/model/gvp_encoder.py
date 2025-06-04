# src/model/gvp_encoder.py
import torch
import torch.nn as nn
from src.gvp import GVPConvLayer, LayerNorm, Dropout, GVP

class GVPGraphEncoder(nn.Module):
    """
    GVP Graph Encoder that safely handles zero vector dimensions.
    Uses different architectures for vector vs scalar-only cases.
    """
    def __init__(self, 
                 node_dims=(6, 3),         # (scalar, vector)
                 edge_dims=(32, 1),        # (scalar, vector)
                 hidden_dims=(128, 16),    # (scalar, vector) hidden dims
                 num_layers=3,
                 drop_rate=0.1):
        super().__init__()
        
        self.node_dims = node_dims
        self.edge_dims = edge_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        
        # Check if this is a zero-vector case
        self.has_node_vectors = node_dims[1] > 0 and hidden_dims[1] > 0
        self.has_edge_vectors = edge_dims[1] > 0
        
        if self.has_node_vectors and self.has_edge_vectors:
            # Full GVP case: use normal GVP layers
            self._build_gvp_layers(drop_rate)
        else:
            # Zero vector case: use scalar-only layers
            self._build_scalar_layers(drop_rate)
    
    def _build_gvp_layers(self, drop_rate):
        """Build full GVP layers for cases with vector features"""
        self.input_proj = GVP(self.node_dims, self.hidden_dims)
        
        self.layers = nn.ModuleList([
            GVPConvLayer(self.hidden_dims, self.edge_dims, 
                         drop_rate=drop_rate)
            for _ in range(self.num_layers)
        ])
        
        self.norm = LayerNorm(self.hidden_dims)
        self.dropout = Dropout(drop_rate)
        self.use_gvp = True
    
    def _build_scalar_layers(self, drop_rate):
        """Build scalar-only layers for zero vector cases"""
        from torch_geometric.nn import GCNConv, global_mean_pool
        
        # Input projection
        self.input_proj = nn.Linear(self.node_dims[0], self.hidden_dims[0])
        
        # Use standard graph conv layers instead of GVP
        self.layers = nn.ModuleList([
            nn.Sequential(
                GCNConv(self.hidden_dims[0], self.hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(drop_rate)
            )
            for _ in range(self.num_layers)
        ])
        
        self.norm = nn.LayerNorm(self.hidden_dims[0])
        self.dropout = nn.Dropout(drop_rate)
        self.use_gvp = False
    
    def forward(self, x_s, x_v, edge_index, edge_s, edge_v):
        """
        Forward pass that handles both GVP and scalar-only cases
        """
        if self.use_gvp:
            # Full GVP forward pass
            x = (x_s, x_v)
            x = self.input_proj(x)
            edge_attr = (edge_s, edge_v)
            
            for layer in self.layers:
                x = layer(x, edge_index, edge_attr)
            
            x = self.norm(self.dropout(x))
            return x
        
        else:
            # Scalar-only forward pass
            x_s_proj = self.input_proj(x_s)
            
            for layer in self.layers:
                x_s_proj = layer(x_s_proj, edge_index)
            
            x_s_final = self.norm(self.dropout(x_s_proj))
            
            # Create dummy vector output to maintain tuple format
            x_v_dummy = torch.zeros(x_s_final.size(0), self.hidden_dims[1], 3,
                                  device=x_s_final.device, dtype=x_s_final.dtype)
            
            return (x_s_final, x_v_dummy)


# Alternative: Hybrid approach that uses minimal vector dimensions
class GVPGraphEncoderHybrid(nn.Module):
    """
    Hybrid encoder that converts zero-vector cases to minimal vectors
    to avoid GVP issues while still using the GVP architecture
    """
    def __init__(self, 
                 node_dims=(6, 3),         
                 edge_dims=(32, 1),        
                 hidden_dims=(128, 16),    
                 num_layers=3,
                 drop_rate=0.1):
        super().__init__()
        
        self.original_node_dims = node_dims
        self.original_edge_dims = edge_dims
        self.original_hidden_dims = hidden_dims
        
        # Convert zero dimensions to minimal dimensions for GVP compatibility
        self.gvp_node_dims = (node_dims[0], max(1, node_dims[1]))
        self.gvp_edge_dims = (edge_dims[0], max(1, edge_dims[1]))  
        self.gvp_hidden_dims = (hidden_dims[0], max(1, hidden_dims[1]))
        
        # Build with minimal dimensions
        self.input_proj = GVP(self.gvp_node_dims, self.gvp_hidden_dims)
        
        self.layers = nn.ModuleList([
            GVPConvLayer(self.gvp_hidden_dims, self.gvp_edge_dims,
                         drop_rate=drop_rate)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(self.gvp_hidden_dims)
        self.dropout = Dropout(drop_rate)
        
        # Track which dimensions were originally zero
        self.node_vectors_were_zero = node_dims[1] == 0
        self.edge_vectors_were_zero = edge_dims[1] == 0
        self.output_vectors_should_be_zero = hidden_dims[1] == 0
    
    def forward(self, x_s, x_v, edge_index, edge_s, edge_v):
        # Convert zero vector inputs to minimal vectors
        if self.node_vectors_were_zero:
            x_v = torch.zeros(x_s.size(0), 1, 3, device=x_s.device, dtype=x_s.dtype)
        
        if self.edge_vectors_were_zero:
            edge_v = torch.zeros(edge_s.size(0), 1, 3, device=edge_s.device, dtype=edge_s.dtype)
        
        # Normal GVP forward pass
        x = (x_s, x_v)
        x = self.input_proj(x)
        edge_attr = (edge_s, edge_v)
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        x = self.norm(self.dropout(x))
        
        # Convert output back to zero vectors if needed
        if self.output_vectors_should_be_zero:
            x_v_zero = torch.zeros(x[0].size(0), 0, 3, device=x[0].device, dtype=x[0].dtype)
            x = (x[0], x_v_zero)
        
        return x


# Test both approaches
def test_zero_vector_solutions():
    """Test both solutions for zero vector handling"""
    print("=== Testing Zero Vector Solutions ===")
    
    # Test data for ligand (zero vectors)
    batch_size = 20
    num_edges = 40
    x_s = torch.randn(batch_size, 46)
    x_v = torch.zeros(batch_size, 0, 3)
    edge_s = torch.randn(num_edges, 9)
    edge_v = torch.zeros(num_edges, 0, 3)
    edge_index = torch.randint(0, batch_size, (2, num_edges))
    
    print(f"Input shapes: x_s={x_s.shape}, x_v={x_v.shape}")
    print(f"Edge shapes: edge_s={edge_s.shape}, edge_v={edge_v.shape}")
    
    # Test 1: Safe encoder (uses scalar layers for zero vector case)
    print(f"\n--- Test 1: Safe Encoder ---")
    try:
        encoder1 = GVPGraphEncoder(
            node_dims=(46, 0), edge_dims=(9, 0), hidden_dims=(128, 0), num_layers=2
        )
        out1 = encoder1(x_s, x_v, edge_index, edge_s, edge_v)
        print(f"✓ Safe encoder: s={out1[0].shape}, v={out1[1].shape}")
    except Exception as e:
        print(f"✗ Safe encoder failed: {e}")
    
    # Test 2: Hybrid encoder (uses minimal vectors)
    print(f"\n--- Test 2: Hybrid Encoder ---") 
    try:
        encoder2 = GVPGraphEncoderHybrid(
            node_dims=(46, 0), edge_dims=(9, 0), hidden_dims=(128, 0), num_layers=2
        )
        out2 = encoder2(x_s, x_v, edge_index, edge_s, edge_v)
        print(f"✓ Hybrid encoder: s={out2[0].shape}, v={out2[1].shape}")
    except Exception as e:
        print(f"✗ Hybrid encoder failed: {e}")

if __name__ == "__main__":
    test_zero_vector_solutions()