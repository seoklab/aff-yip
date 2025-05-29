import torch
import torch.nn as nn
import torch.nn.functional as F

def norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """Computes the norm of tensor x, avoiding NaN gradients."""
    out = torch.clamp(torch.sum(torch.square(x), dim=axis, keepdim=keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

def split(x, nv):
    """
    Splits concatenated vector and scalar features
    :param x: combined features [batch, *, 3*nv + ns]
    :param nv: number of vector features
    :return: tuple (vector, scalar) features
    """
    v = x[..., :3*nv].reshape(x.shape[:-1] + (3, nv))
    s = x[..., 3*nv:]
    return v, s

def merge(v, s):
    """
    Combines vector and scalar features
    :param v: vector features [batch, *, 3, nv]
    :param s: scalar features [batch, *, ns]
    :return: combined features [batch, *, 3*nv + ns]
    """
    v = v.reshape(v.shape[:-2] + (3*v.shape[-1],))
    return torch.cat([v, s], dim=-1)

def vs_concat(x1, x2, nv1, nv2):
    """
    Concatenates two sets of features along the feature dimension
    while keeping the vector and scalar parts separate.
    """
    v1, s1 = split(x1, nv1)
    v2, s2 = split(x2, nv2)
    
    v = torch.cat([v1, v2], dim=-1)
    s = torch.cat([s1, s2], dim=-1)
    return merge(v, s)

class GVP(nn.Module):
    """
    Geometric Vector Perceptron. Transforms vector and scalar features
    :param vi: input vector channels
    :param vo: output vector channels
    :param so: output scalar channels
    :param nlv: nonlinearity for vector features
    :param nls: nonlinearity for scalar features
    """
    def __init__(self, vi, vo, so, 
                 nlv=torch.sigmoid, nls=F.relu):
        """[v/s][i/o] = number of [vector/scalar] channels [in/out]"""
        super(GVP, self).__init__()
        self.vi, self.vo, self.so = vi, vo, so
        self.nlv, self.nls = nlv, nls
        
        if vi:
            self.wh = nn.Linear(vi, max(vi, vo))
        if vo:
            self.wv = nn.Linear(max(vi, vo), vo)
        self.ws = nn.Linear(vi+so if vi else so, so)
        
        if nls is not None:
            self.ns = nls
    
    def forward(self, x, return_split=False):
        """
        :param x: tuple (v, s) of vector and scalar features
            or combined features [batch, ..., 3*vi + si]
        :param return_split: whether to return split vector and scalar features
        :return: tuple (v, s) if return_split, else combined features
        """
        if not isinstance(x, tuple):
            v, s = split(x, self.vi)
        else:
            v, s = x
            
        if self.vi:
            vh = self.wh(v)
            vn = norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], dim=-1))
        else:
            s = self.ws(s)
            
        if self.nls is not None:
            s = self.nls(s)
            
        if self.vo:
            vo = self.wv(vh)
            if self.nlv is not None:
                vo = vo * self.nlv(norm_no_nan(vo, axis=-2, keepdims=True))
            if return_split:
                return vo, s
            return merge(vo, s)
        
        if return_split:
            return None, s
        return s


class GVPDropout(nn.Module):
    """
    Dropout layer for GVP features. Applies dropout separately to vector and scalar channels.
    """
    def __init__(self, rate, nv):
        super(GVPDropout, self).__init__()
        self.nv = nv
        self.vdropout = nn.Dropout(rate)
        self.sdropout = nn.Dropout(rate)
    
    def forward(self, x):
        """
        :param x: combined features [batch, ..., 3*nv + ns]
        :return: features with dropout applied
        """
        if not self.training:
            return x
        
        v, s = split(x, self.nv)
        
        # Create binary dropout masks - need to repeat across the vector dimension for vectors
        v = self.vdropout(v)
        s = self.sdropout(s)
        
        return merge(v, s)


class GVPLayerNorm(nn.Module):
    """
    Layer normalization for GVP features.
    Normalizes scalar features with learned parameters and
    normalizes vector features to unit norm.
    """
    def __init__(self, nv):
        super(GVPLayerNorm, self).__init__()
        self.nv = nv
        self.snorm = nn.LayerNorm(normalized_shape=[])  # Will be resized during forward pass
    
    def forward(self, x):
        """
        :param x: combined features [batch, ..., 3*nv + ns]
        :return: normalized features
        """
        v, s = split(x, self.nv)
        
        # Normalize vector magnitudes
        vn = norm_no_nan(v, axis=-2, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-1, keepdim=True))
        v = v / vn
        
        # Normalize scalar features
        if s.size(-1) > 0:  # Only apply if we have scalar features
            if self.snorm.normalized_shape != list(s.shape[-1:]):
                self.snorm.normalized_shape = list(s.shape[-1:])
                self.snorm.reset_parameters()
            s = self.snorm(s)
        
        return merge(v, s)

# Aliases for compatibility with the original code
Velu = GVP
VGDropout = GVPDropout
VGLayerNorm = GVPLayerNorm