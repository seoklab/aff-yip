# integrated_flow_matching.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional

class FlowMatchingStructurePredictor(nn.Module):
    
    def __init__(self,
                 enhanced_virtual_embed_dim: int = 196,  # interaction_output dim
                 ligand_embed_dim: int = 196,            # interaction_output dim
                 hidden_dim: int = 256,
                 num_timesteps: int = 1000,
                 num_flow_layers: int = 4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        
        # Flow matching schedule
        self.register_buffer('betas', self._cosine_beta_schedule(num_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - self.alphas_cumprod))
        
        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        
        # === PROCESS YOUR INTERACTION RESULTS ===
        # Project your enhanced embeddings from the interaction model
        self.enhanced_virtual_proj = nn.Linear(enhanced_virtual_embed_dim, hidden_dim)
        self.ligand_embed_proj = nn.Linear(ligand_embed_dim, hidden_dim)
        
        # Current coordinate embedding (during denoising)
        self.coord_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # === FLOW DENOISING NETWORK ===
        self.flow_layers = nn.ModuleList([
            FlowDenoisingLayer(hidden_dim) for _ in range(num_flow_layers)
        ])
        
        # Final velocity prediction
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)  # Predict 3D velocity
        )
        
    def _cosine_beta_schedule(self, timesteps):
        """Cosine schedule from FlowSite"""
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 1e-4, 0.02)
    
    def add_noise(self, clean_coords, t, noise=None):
        """Add noise for flow matching training"""
        if noise is None:
            noise = torch.randn_like(clean_coords)
        
        # Handle batch-level timesteps
        batch_indices = torch.arange(clean_coords.shape[0], device=clean_coords.device)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t[batch_indices % len(t)]]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t[batch_indices % len(t)]]
        
        # Expand to coordinate dimensions
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        noisy_coords = sqrt_alphas_cumprod_t * clean_coords + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_coords, noise
    
    def create_conditioning_from_interaction_results(self, interaction_results):
        """
        Convert YOUR interaction_results into conditioning for flow matching
        
        Args:
            interaction_results: Dict from your get_embeddings() containing:
                - enhanced_virtual_embeddings: [N_virtual, 196] 
                - ligand_embeddings: [N_ligand, 196]
                - virtual_batch_idx, ligand_batch_idx, etc.
        """
        # Project your enhanced embeddings
        enhanced_virtual_context = self.enhanced_virtual_proj(
            interaction_results['enhanced_virtual_embeddings']
        )  # [N_virtual, hidden_dim]
        
        ligand_context = self.ligand_embed_proj(
            interaction_results['ligand_embeddings']  
        )  # [N_ligand, hidden_dim]
        
        return {
            'virtual_context': enhanced_virtual_context,
            'ligand_context': ligand_context,
            'virtual_batch_idx': interaction_results['virtual_batch_idx'],
            'ligand_batch_idx': interaction_results['ligand_batch_idx'],
            'virtual_coords': interaction_results['virtual_coords']  # For reference
        }
    
    def denoise_step(self, noisy_coords, t, conditioning, target_mask):
        """
        Single flow matching denoising step
        
        Args:
            noisy_coords: [batch_size, max_atoms, 3] - current noisy coordinates
            t: [batch_size] - timestep
            conditioning: processed interaction results
            target_mask: [batch_size, max_atoms] - valid atom mask
        """
        batch_size, max_atoms, _ = noisy_coords.shape
        device = noisy_coords.device
        
        # Time embedding
        t_embed = self.time_embedding(t)  # [batch_size, hidden_dim]
        t_embed = t_embed.unsqueeze(1).expand(-1, max_atoms, -1)  # [batch_size, max_atoms, hidden_dim]
        
        # Current coordinate embedding
        coord_embed = self.coord_embedding(noisy_coords)  # [batch_size, max_atoms, hidden_dim]
        
        # Combine coordinate and time features
        atom_features = coord_embed + t_embed
        
        # Add ligand context to each atom
        ligand_context = conditioning['ligand_context']  # [N_ligand, hidden_dim]
        ligand_batch_idx = conditioning['ligand_batch_idx']
        
        # Map ligand context to batched format
        ligand_features_batched = torch.zeros(batch_size, max_atoms, self.hidden_dim, device=device)
        
        for b in range(batch_size):
            ligand_mask = ligand_batch_idx == b
            if ligand_mask.any():
                num_ligand_atoms = ligand_mask.sum().item()
                ligand_features_batched[b, :num_ligand_atoms] = ligand_context[ligand_mask]
        
        # Combine atom features with ligand context
        atom_features = atom_features + ligand_features_batched
        
        # Apply flow denoising layers with virtual node conditioning
        for flow_layer in self.flow_layers:
            atom_features = flow_layer(
                atom_features=atom_features,
                coords=noisy_coords,
                virtual_context=conditioning['virtual_context'],
                virtual_batch_idx=conditioning['virtual_batch_idx'],
                target_mask=target_mask
            )
        
        # Predict velocity
        velocity = self.velocity_head(atom_features)  # [batch_size, max_atoms, 3]
        
        # Mask out invalid atoms
        velocity = velocity * target_mask.unsqueeze(-1)
        
        return velocity
    
    def sample_harmonic_prior(self, target_shape, ligand_batch_idx, device):
        """
        Sample from harmonic prior (can be enhanced with bond structure later)
        """
        batch_size, max_atoms, _ = target_shape
        
        # For now, sample from Gaussian (TODO: use ligand bond structure)
        coords = torch.randn(batch_size, max_atoms, 3, device=device)
        
        # Could enhance this by:
        # 1. Using ligand bond graph to create harmonic potential
        # 2. Sampling connected atoms closer together
        # 3. Using virtual node positions as centers
        
        return coords
    
    @torch.no_grad()
    def sample(self, interaction_results, target_mask, num_steps=20):
        """
        Generate coordinates using flow matching
        
        Args:
            interaction_results: YOUR interaction results from get_embeddings()
            target_mask: [batch_size, max_atoms] - which atoms to generate
            num_steps: number of integration steps
        """
        batch_size, max_atoms = target_mask.shape
        device = target_mask.device
        
        # Create conditioning from your interaction results
        conditioning = self.create_conditioning_from_interaction_results(interaction_results)
        
        # Initialize from harmonic prior
        coords = self.sample_harmonic_prior(
            target_shape=(batch_size, max_atoms, 3),
            ligand_batch_idx=conditioning['ligand_batch_idx'], 
            device=device
        )
        
        # Mask initial coordinates
        coords = coords * target_mask.unsqueeze(-1)
        
        # Flow matching integration
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=device)
            
            # Predict velocity using your enhanced embeddings
            velocity = self.denoise_step(coords, t, conditioning, target_mask)
            
            # Euler integration step
            coords = coords + dt * velocity
            
            # Ensure valid atoms stay valid
            coords = coords * target_mask.unsqueeze(-1)
        
        return coords
    
    def forward(self, interaction_results, target_mask, target_coords=None, training=True):
        """
        Main forward pass using YOUR interaction results
        
        Args:
            interaction_results: Dict from your get_embeddings() method
            target_mask: [batch_size, max_atoms] - valid atom mask
            target_coords: [batch_size, max_atoms, 3] - ground truth (training only)
            training: whether in training mode
        """
        batch_size, max_atoms = target_mask.shape
        device = target_mask.device
        
        # Create conditioning from your interaction results
        conditioning = self.create_conditioning_from_interaction_results(interaction_results)
        
        if training and target_coords is not None:
            # Training: predict velocity for flow matching
            
            # Sample timestep
            t = torch.rand(batch_size, device=device)  # [0, 1]
            
            # Sample from harmonic prior
            prior_coords = self.sample_harmonic_prior(
                target_shape=(batch_size, max_atoms, 3),
                ligand_batch_idx=conditioning['ligand_batch_idx'],
                device=device
            )
            
            # Linear interpolation (flow matching)
            t_expanded = t.view(batch_size, 1, 1)  # [batch_size, 1, 1]
            noisy_coords = t_expanded * target_coords + (1 - t_expanded) * prior_coords
            
            # Mask coordinates
            noisy_coords = noisy_coords * target_mask.unsqueeze(-1)
            target_coords = target_coords * target_mask.unsqueeze(-1)
            target_coords_masked = target_coords * target_mask.unsqueeze(-1)
            prior_coords = prior_coords * target_mask.unsqueeze(-1)
            
            # True velocity is (target - prior)
            true_velocity = target_coords - prior_coords
            
            # Predict velocity using your enhanced embeddings
            pred_velocity = self.denoise_step(noisy_coords, t, conditioning, target_mask)
            # Predicted coordinates from velocity
            pred_coords = prior_coords + pred_velocity

            return pred_velocity, true_velocity, pred_coords, target_coords_masked
        else:
            # Inference: generate coordinates
            generated_coords = self.sample(interaction_results, target_mask, num_steps=20)
            return generated_coords, None, generated_coords, None  # No true coords in inference


class FlowDenoisingLayer(nn.Module):
    """
    Single layer of the flow denoising network
    Processes atoms with virtual node conditioning
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Self-attention among atoms
        self.atom_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Cross-attention with virtual nodes (your enhanced embeddings!)
        self.virtual_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self, atom_features, coords, virtual_context, virtual_batch_idx, target_mask):
        """
        Args:
            atom_features: [batch_size, max_atoms, hidden_dim]
            coords: [batch_size, max_atoms, 3] - current coordinates
            virtual_context: [N_virtual, hidden_dim] - YOUR enhanced virtual embeddings
            virtual_batch_idx: [N_virtual] - batch assignment
            target_mask: [batch_size, max_atoms] - valid atoms
        """
        batch_size, max_atoms, hidden_dim = atom_features.shape
        
        # Self-attention among atoms
        residual = atom_features
        atom_features_flat = atom_features.view(batch_size, max_atoms, hidden_dim)
        
        # Create attention mask for padding
        attn_mask = ~target_mask  # True for padded positions
        
        attended, _ = self.atom_self_attention(
            atom_features_flat, atom_features_flat, atom_features_flat,
            key_padding_mask=attn_mask
        )
        atom_features = self.norm1(attended + residual)
        
        # Cross-attention with YOUR virtual node embeddings
        residual = atom_features
        atom_features = self._apply_virtual_cross_attention(
            atom_features, virtual_context, virtual_batch_idx, target_mask
        )
        atom_features = self.norm2(atom_features + residual)
        
        # Feed-forward
        residual = atom_features
        atom_features = self.ffn(atom_features)
        atom_features = self.norm3(atom_features + residual)
        
        return atom_features
    
    def _apply_virtual_cross_attention(self, atom_features, virtual_context, virtual_batch_idx, target_mask):
        """Apply cross-attention between atoms and YOUR virtual embeddings"""
        batch_size, max_atoms, hidden_dim = atom_features.shape
        device = atom_features.device
        
        output = torch.zeros_like(atom_features)
        
        for b in range(batch_size):
            # Get atoms for this batch
            atom_mask = target_mask[b]  # [max_atoms]
            if not atom_mask.any():
                continue
            
            atoms_b = atom_features[b:b+1, atom_mask]  # [1, N_atoms_b, hidden_dim]
            
            # Get virtual nodes for this batch
            virtual_mask = virtual_batch_idx == b
            if virtual_mask.any():
                virtual_b = virtual_context[virtual_mask].unsqueeze(0)  # [1, N_virtual_b, hidden_dim]
                
                # Cross-attention: atoms attend to YOUR virtual embeddings
                attended, _ = self.virtual_cross_attention(atoms_b, virtual_b, virtual_b)
                output[b, atom_mask] = attended.squeeze(0)
            else:
                # No virtual nodes, keep original features
                output[b, atom_mask] = atoms_b.squeeze(0)
        
        return output


class SinusoidalTimeEmbedding(nn.Module):
    """Time embedding for flow matching"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

