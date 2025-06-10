# src/model/gvp_encoder.py
import torch
import torch.nn as nn
from src.gvp import GVPConvLayer, LayerNorm, Dropout, GVP
from typing import Tuple, Optional, Union

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


class GVPGraphEncoderHybrid(nn.Module):
    """
    Enhanced hybrid encoder that handles zero-vector cases with improved efficiency
    and better error handling for GVP architectures.
    
    Key improvements over the original:
    - More robust dimension validation
    - Better memory efficiency for zero vector cases
    - Cleaner output dimension handling
    - Optional minimal vector injection for stability
    """
    def __init__(self, 
                 node_dims: Tuple[int, int] = (6, 3),         
                 edge_dims: Tuple[int, int] = (32, 1),        
                 hidden_dims: Tuple[int, int] = (128, 16),    
                 num_layers: int = 3,
                 drop_rate: float = 0.1,
                 minimal_vector_noise: float = 1e-6,
                 preserve_zero_output: bool = True):
        """
        Args:
            node_dims: (scalar_features, vector_features) for nodes
            edge_dims: (scalar_features, vector_features) for edges  
            hidden_dims: (scalar_features, vector_features) for hidden layers
            num_layers: number of GVP conv layers
            drop_rate: dropout probability
            minimal_vector_noise: small noise added to minimal vectors for numerical stability
            preserve_zero_output: whether to return zero vectors when input had zero vectors
        """
        super().__init__()

        self.original_node_dims = node_dims
        self.original_edge_dims = edge_dims
        self.original_hidden_dims = hidden_dims
        self.minimal_vector_noise = minimal_vector_noise
        self.preserve_zero_output = preserve_zero_output
        
        # Convert zero dimensions to minimal dimensions for GVP compatibility
        self.gvp_node_dims = (node_dims[0], max(1, node_dims[1]))
        self.gvp_edge_dims = (edge_dims[0], max(1, edge_dims[1]))  
        self.gvp_hidden_dims = (hidden_dims[0], max(1, hidden_dims[1]))
        
        # Track which dimensions were originally zero for efficient processing
        self.node_vectors_zero = node_dims[1] == 0
        self.edge_vectors_zero = edge_dims[1] == 0
        self.hidden_vectors_zero = hidden_dims[1] == 0
        
        # Build network with GVP-compatible dimensions
        self.input_proj = GVP(self.gvp_node_dims, self.gvp_hidden_dims)
        
        self.layers = nn.ModuleList([
            GVPConvLayer(
                self.gvp_hidden_dims, 
                self.gvp_edge_dims,
                drop_rate=drop_rate
            )
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(self.gvp_hidden_dims)
        self.dropout = Dropout(drop_rate)
        
        # Cache for minimal vectors to avoid repeated allocation
        self._node_minimal_cache = None
        self._edge_minimal_cache = None
    
    def _get_minimal_vectors(self, 
                           batch_size: int, 
                           vector_dim: int, 
                           device: torch.device, 
                           dtype: torch.dtype,
                           cache_key: str) -> torch.Tensor:
        """
        Get minimal vectors with optional caching and small noise for numerical stability.
        """
        cache_attr = f"_{cache_key}_minimal_cache"
        cached = getattr(self, cache_attr, None)
        
        # Check if we can reuse cached tensor
        if (cached is not None and 
            cached.shape[0] >= batch_size and 
            cached.device == device and 
            cached.dtype == dtype):
            result = cached[:batch_size]
        else:
            # Create new minimal vectors
            if self.minimal_vector_noise > 0:
                result = torch.randn(batch_size, vector_dim, 3, 
                                   device=device, dtype=dtype) * self.minimal_vector_noise
            else:
                result = torch.zeros(batch_size, vector_dim, 3, 
                                   device=device, dtype=dtype)
            
            # Update cache if beneficial (avoid caching very large tensors)
            if batch_size <= 10000:  # reasonable cache size limit
                setattr(self, cache_attr, result.clone())
        
        return result
    
    def _prepare_inputs(self, x_s, x_v, edge_s, edge_v):
        """
        Prepare inputs by converting zero-dimensional vectors to minimal vectors.
        """
        # Handle node vectors
        if x_v.size(1) == 0:  # Runtime check is more robust
            x_v = self._get_minimal_vectors(
                x_s.size(0), 1, x_s.device, x_s.dtype, "node"
            )
        
        # Handle edge vectors  
        if edge_v.size(1) == 0:
            edge_v = self._get_minimal_vectors(
                edge_s.size(0), 1, edge_s.device, edge_s.dtype, "edge"
            )
            
        return x_v, edge_v
    
    def _prepare_output(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare output by converting back to zero vectors if needed.
        """
        x_s, x_v = x
        
        if self.preserve_zero_output and self.hidden_vectors_zero:
            # Return zero-dimensional vectors to match expected output
            x_v_zero = torch.zeros(x_s.size(0), 0, 3, device=x_s.device, dtype=x_s.dtype)
            return (x_s, x_v_zero)
        
        return (x_s, x_v)
    
    def forward(self, 
                x_s: torch.Tensor, 
                x_v: torch.Tensor, 
                edge_index: torch.Tensor, 
                edge_s: torch.Tensor, 
                edge_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with automatic zero vector handling.
        
        Args:
            x_s: Node scalar features [N, node_scalar_dim]
            x_v: Node vector features [N, node_vector_dim, 3] 
            edge_index: Edge connectivity [2, E]
            edge_s: Edge scalar features [E, edge_scalar_dim]
            edge_v: Edge vector features [E, edge_vector_dim, 3]
            
        Returns:
            Tuple of (scalar_features, vector_features)
        """
        # Validate input shapes
        assert x_s.dim() == 2, f"Expected x_s to be 2D, got {x_s.dim()}D"
        assert x_v.dim() == 3, f"Expected x_v to be 3D, got {x_v.dim()}D"
        assert x_v.size(-1) == 3, f"Expected last dim of x_v to be 3, got {x_v.size(-1)}"
        
        # Prepare inputs - convert zero vectors to minimal vectors
        x_v_processed, edge_v_processed = self._prepare_inputs(x_s, x_v, edge_s, edge_v)
        
        # Standard GVP forward pass
        x = (x_s, x_v_processed)
        x = self.input_proj(x)
        edge_attr = (edge_s, edge_v_processed)
        
        # Apply GVP convolution layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # Final normalization and dropout
        x = self.norm(self.dropout(x))
        
        # Prepare output - convert back to zero vectors if needed
        return self._prepare_output(x)
    
    def get_output_dims(self) -> Tuple[int, int]:
        """Get the actual output dimensions (accounting for zero vector conversion)."""
        if self.preserve_zero_output and self.hidden_vectors_zero:
            return (self.original_hidden_dims[0], 0)
        return self.gvp_hidden_dims
    
    def reset_cache(self):
        """Reset cached minimal vectors (useful for memory management)."""
        self._node_minimal_cache = None
        self._edge_minimal_cache = None


class GVPGraphEncoderSafe(nn.Module):
    """
    Alternative safe encoder that uses pure scalar layers when vectors are zero.
    This approach completely avoids GVP layers for zero-vector cases.
    """
    def __init__(self, 
                 node_dims: Tuple[int, int] = (6, 3),         
                 edge_dims: Tuple[int, int] = (32, 1),        
                 hidden_dims: Tuple[int, int] = (128, 16),    
                 num_layers: int = 3,
                 drop_rate: float = 0.1):
        super().__init__()
        
        self.node_dims = node_dims
        self.edge_dims = edge_dims  
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        
        # Check if we have any vector dimensions
        self.has_vectors = (node_dims[1] > 0 or edge_dims[1] > 0 or hidden_dims[1] > 0)
        
        if self.has_vectors:
            # Use standard GVP layers
            self.input_proj = GVP(node_dims, hidden_dims)
            self.layers = nn.ModuleList([
                GVPConvLayer(hidden_dims, edge_dims, drop_rate=drop_rate)
                for _ in range(num_layers)
            ])
            self.norm = LayerNorm(hidden_dims)
        else:
            # Use pure scalar layers - much more efficient for zero vector case
            from torch_geometric.nn import GCNConv
            
            self.input_proj = nn.Linear(node_dims[0], hidden_dims[0])
            self.layers = nn.ModuleList([
                GCNConv(hidden_dims[0], hidden_dims[0])
                for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(hidden_dims[0])
            
        self.dropout = nn.Dropout(drop_rate)
        self.activation = nn.ReLU()
    
    def forward(self, x_s, x_v, edge_index, edge_s, edge_v):
        if self.has_vectors:
            # Standard GVP path
            x = (x_s, x_v)
            x = self.input_proj(x)
            edge_attr = (edge_s, edge_v)
            
            for layer in self.layers:
                x = layer(x, edge_index, edge_attr)
            
            x = self.norm(self.dropout(x))
            return x
        else:
            # Efficient scalar-only path
            x = self.input_proj(x_s)
            x = self.activation(x)
            
            for layer in self.layers:
                x = layer(x, edge_index, edge_weight=edge_s.norm(dim=-1))
                x = self.activation(x)
            
            x = self.norm(self.dropout(x))
            
            # Return in tuple format for consistency
            zero_vectors = torch.zeros(x.size(0), 0, 3, device=x.device, dtype=x.dtype)
            return (x, zero_vectors)


def test_zero_vector_solutions():
    """Enhanced test function with more comprehensive validation."""
    print("=== Testing Enhanced Zero Vector Solutions ===")
    
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
    
    # Test 1: Enhanced Hybrid encoder
    print(f"\n--- Test 1: Enhanced Hybrid Encoder ---")
    try:
        encoder1 = GVPGraphEncoderHybrid(
            node_dims=(46, 0), 
            edge_dims=(9, 0), 
            hidden_dims=(128, 0), 
            num_layers=2,
            preserve_zero_output=True
        )
        out1 = encoder1(x_s, x_v, edge_index, edge_s, edge_v)
        print(f"✓ Enhanced hybrid encoder: s={out1[0].shape}, v={out1[1].shape}")
        print(f"  Output dims: {encoder1.get_output_dims()}")
        
        # Test memory efficiency
        encoder1.reset_cache()
        print(f"  Cache reset successful")
        
    except Exception as e:
        print(f"✗ Enhanced hybrid encoder failed: {e}")
    
    # Test 2: Safe encoder
    print(f"\n--- Test 2: Safe Encoder (Scalar-only) ---") 
    try:
        encoder2 = GVPGraphEncoderSafe(
            node_dims=(46, 0), 
            edge_dims=(9, 0), 
            hidden_dims=(128, 0), 
            num_layers=2
        )
        out2 = encoder2(x_s, x_v, edge_index, edge_s, edge_v)
        print(f"✓ Safe encoder: s={out2[0].shape}, v={out2[1].shape}")
        print(f"  Using vectors: {encoder2.has_vectors}")
        
    except Exception as e:
        print(f"✗ Safe encoder failed: {e}")
    
    # Test 3: Performance comparison
    print(f"\n--- Test 3: Performance Comparison ---")
    import time
    
    # Warmup
    for encoder in [encoder1, encoder2]:
        for _ in range(5):
            _ = encoder(x_s, x_v, edge_index, edge_s, edge_v)
    
    # Timing test
    n_runs = 100
    
    start_time = time.time()
    for _ in range(n_runs):
        _ = encoder1(x_s, x_v, edge_index, edge_s, edge_v)
    hybrid_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(n_runs):
        _ = encoder2(x_s, x_v, edge_index, edge_s, edge_v)
    safe_time = time.time() - start_time
    
    print(f"Hybrid encoder: {hybrid_time:.4f}s ({hybrid_time/n_runs*1000:.2f}ms per run)")
    print(f"Safe encoder: {safe_time:.4f}s ({safe_time/n_runs*1000:.2f}ms per run)")
    print(f"Safe encoder is {hybrid_time/safe_time:.1f}x faster")

if __name__ == "__main__":
    test_zero_vector_solutions()
