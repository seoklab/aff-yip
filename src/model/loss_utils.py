import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateLoss(nn.Module):
    """Loss function for coordinate prediction with masking support"""
    
    def __init__(self, loss_type="mse", reduction="mean"):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
    
    def compute_rmsd_for_logging(self, pred_coords, target_coords, mask):
        """
        Compute RMSD for logging purposes (separate from loss)
        
        Args:
            pred_coords: [B, max_atoms, 3] predicted coordinates
            target_coords: [B, max_atoms, 3] ground truth coordinates
            mask: [B, max_atoms] boolean mask for valid atoms
        
        Returns:
            rmsd: Scalar RMSD value for logging
        """
        batch_size = pred_coords.size(0)
        molecule_rmsds = []
        
        for b in range(batch_size):
            # Get valid atoms for this molecule
            valid_atoms = mask[b]  # [max_atoms]
            num_valid = valid_atoms.sum().item()
            
            if num_valid == 0:
                continue
            
            # Calculate RMSD for this molecule
            pred_mol = pred_coords[b][valid_atoms]    # [num_valid, 3]
            target_mol = target_coords[b][valid_atoms]  # [num_valid, 3]
            
            # RMSD = sqrt(mean(||pred_atom - target_atom||^2))
            atom_squared_distances = ((pred_mol - target_mol) ** 2).sum(dim=-1)  # [num_valid]
            molecule_rmsd = torch.sqrt(atom_squared_distances.mean())
            molecule_rmsds.append(molecule_rmsd)
        
        if molecule_rmsds:
            return torch.stack(molecule_rmsds).mean()
        else:
            return torch.tensor(0.0, device=pred_coords.device)
    
    def forward(self, pred_coords, target_coords, mask):
        """
        Args:
            pred_coords: [B, max_atoms, 3] predicted coordinates
            target_coords: [B, max_atoms, 3] ground truth coordinates  
            mask: [B, max_atoms] boolean mask for valid atoms
        
        Returns:
            loss: scalar coordinate loss
        """
        # Expand mask to 3D coordinates
        mask_3d = mask.unsqueeze(-1).expand_as(pred_coords)  # [B, max_atoms, 3]
        
        if self.loss_type == "mse":
            # Masked MSE loss
            diff_squared = (pred_coords - target_coords) ** 2
            masked_diff = diff_squared * mask_3d.float()
            
            if self.reduction == "mean":
                loss = masked_diff.sum() / (mask_3d.float().sum() + 1e-8)
            elif self.reduction == "sum":
                loss = masked_diff.sum()
            else:
                loss = masked_diff
                
        elif self.loss_type == "mae":
            # Masked MAE loss
            diff_abs = torch.abs(pred_coords - target_coords)
            masked_diff = diff_abs * mask_3d.float()
            
            if self.reduction == "mean":
                loss = masked_diff.sum() / (mask_3d.float().sum() + 1e-8)
            elif self.reduction == "sum":
                loss = masked_diff.sum()
            else:
                loss = masked_diff
                
        elif self.loss_type == "huber":
            # Masked Huber loss (smooth L1)
            diff = pred_coords - target_coords
            huber_loss = F.smooth_l1_loss(pred_coords, target_coords, reduction='none')
            masked_diff = huber_loss * mask_3d.float()
            
            if self.reduction == "mean":
                loss = masked_diff.sum() / (mask_3d.float().sum() + 1e-8)
            elif self.reduction == "sum":
                loss = masked_diff.sum()
            else:
                loss = masked_diff
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Compute RMSD for logging
        rmsd = self.compute_rmsd_for_logging(pred_coords, target_coords, mask)
        
        return rmsd, loss

class MultiTaskAffinityCoordinateLoss(nn.Module):
    """Combined loss for affinity and coordinate prediction"""
    
    def __init__(self, 
                 affinity_loss_fn,           # Your existing affinity loss (e.g., WeightedCategoryLoss_v2)
                 coord_loss_weight=0.1,      # Weight for coordinate loss
                 coord_loss_type="mse",# Type of coordinate loss
                 lig_internal_dist_w=0.1):     
        super().__init__()
        
        self.affinity_loss_fn = affinity_loss_fn
        self.coord_loss_weight = coord_loss_weight
        if lig_internal_dist_w > 0:
            self.internal_prediction = True
            self.coord_loss = DistanceBasedCoordinateLoss(coord_weight=coord_loss_weight,
                                                           distance_weight=lig_internal_dist_w,
                                                           coord_loss_type=coord_loss_type)
        else:
            self.internal_prediction = False
            self.coord_loss = CoordinateLoss(loss_type=coord_loss_type)
        
    def forward(self, pred_affinity, target_affinity, pred_logits=None, 
                pred_coords=None, target_coords=None, coord_mask=None):
        """
        Args:
            pred_affinity: Predicted affinity values
            target_affinity: True affinity values
            pred_logits: Classification logits (for multitask affinity loss)
            pred_coords: [B, max_atoms, 3] predicted coordinates (optional)
            target_coords: [B, max_atoms, 3] ground truth coordinates (optional)
            coord_mask: [B, max_atoms] mask for valid atoms (optional)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        
        # Affinity loss
        if pred_logits is not None:
            # Multi-task affinity loss (returns multiple components)
            affinity_result = self.affinity_loss_fn(pred_affinity, target_affinity, pred_logits)
            if isinstance(affinity_result, tuple):
                total_affinity_loss, reg_loss, cat_penalty, extreme_penalty, pearson_pen = affinity_result
                loss_dict.update({
                    'affinity_loss': total_affinity_loss,
                    'reg_loss': reg_loss,
                    'category_loss': cat_penalty,
                    'extreme_penalty': extreme_penalty,
                    'ranking_loss': pearson_pen
                })
            else:
                total_affinity_loss = affinity_result
                loss_dict['affinity_loss'] = total_affinity_loss
        else:
            # No category loss
            affinity_result = self.affinity_loss_fn(pred_affinity, target_affinity)
            if isinstance(affinity_result, tuple):
                total_affinity_loss, reg_loss, ranking_loss, _, _ = affinity_result
                loss_dict = {'affinity_loss': total_affinity_loss,
                            'reg_loss': reg_loss, 
                            'ranking_loss': ranking_loss}
        
        total_loss = total_affinity_loss
        
        # Coordinate loss (if provided)
        if pred_coords is not None and target_coords is not None and coord_mask is not None:
            if self.internal_prediction:
                rmsd, coord_loss, str_loss = self.coord_loss(pred_coords, target_coords, coord_mask)
                loss_dict['coord_loss'] = coord_loss # now this is the MSE loss
            else:
                rmsd, str_loss = self.coord_loss(pred_coords, target_coords, coord_mask)
            weighted_str_loss = self.coord_loss_weight * str_loss
            
            total_loss += weighted_str_loss
            loss_dict['ligand_rmsd'] = rmsd
            loss_dict['str_loss'] = weighted_str_loss
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict


class DistanceBasedCoordinateLoss(nn.Module):
    """Coordinate loss that also considers inter-atomic distances"""
    
    def __init__(self, 
                 coord_weight=1.0,
                 distance_weight=0.1,
                 coord_loss_type="mse"):
        super().__init__()
        self.coord_weight = coord_weight
        self.distance_weight = distance_weight
        self.coord_loss = CoordinateLoss(loss_type=coord_loss_type)
        
    def forward(self, pred_coords, target_coords, mask):
        """
        Args:
            pred_coords: [B, max_atoms, 3] predicted coordinates
            target_coords: [B, max_atoms, 3] ground truth coordinates
            mask: [B, max_atoms] boolean mask for valid atoms
        
        Returns:
            total_loss: Combined coordinate + distance loss
        """
        # Basic coordinate loss
        coord_loss, rmsd = self.coord_loss(pred_coords, target_coords, mask)
        total_loss = self.coord_weight * coord_loss
        
        # Distance preservation loss (optional)
        if self.distance_weight > 0:
            distance_loss = self._compute_distance_loss(pred_coords, target_coords, mask)
            total_loss += self.distance_weight * distance_loss
        
        return rmsd, coord_loss, total_loss
    
    def _compute_distance_loss(self, pred_coords, target_coords, mask):
        """Compute loss on pairwise distances between atoms"""
        batch_size, max_atoms, _ = pred_coords.shape
        total_distance_loss = 0.0
        valid_batches = 0
        
        for b in range(batch_size):
            # Get valid atoms for this sample
            valid_mask = mask[b]  # [max_atoms]
            num_valid = valid_mask.sum().item()
            if num_valid <= 1:
                continue  # Need at least 2 atoms for distances
            
            pred_valid = pred_coords[b][valid_mask]      # [num_valid, 3]
            target_valid = target_coords[b][valid_mask]  # [num_valid, 3]
            
            # Compute pairwise distances
            pred_distances = torch.cdist(pred_valid, pred_valid, p=2)      # [num_valid, num_valid]
            target_distances = torch.cdist(target_valid, target_valid, p=2) # [num_valid, num_valid]
            # Only consider upper triangular part (avoid double counting)
            triu_mask = torch.triu(torch.ones_like(pred_distances, dtype=torch.bool), diagonal=1)
            if triu_mask.any():
                distance_diff = (pred_distances[triu_mask] - target_distances[triu_mask]) ** 2
                # print (distance_diff)
                # print ('dist diff',distance_diff.shape) 
                total_distance_loss += distance_diff.mean()
                valid_batches += 1
        
        return total_distance_loss / max(valid_batches, 1)
    
# Drop-in replacement for your current loss
class HuberReplacementLoss(nn.Module):
    """
    Direct replacement for WeightedCategoryLoss_v2 but using Huber loss
    Same interface, much simpler implementation
    """
    def __init__(self,
                 delta=1.0,                    # Huber threshold
                 extreme_weight=2.0,           # Boost for extremes
                 ranking_weight=0.1,           # Preserve correlations
                 # Ignore all the complex threshold parameters
                 **kwargs):                    # Absorb unused parameters
        super().__init__()
        self.delta = delta
        self.extreme_weight = extreme_weight
        self.ranking_weight = ranking_weight
        
    def forward(self, pred_aff, target_aff, pred_logits=None):
        # Main Huber loss
        huber_base = F.smooth_l1_loss(pred_aff, target_aff, beta=self.delta, reduction='none')
        
        # Extreme value emphasis
        extreme_mask = (target_aff < 4.5) | (target_aff > 8.5)
        weights = torch.ones_like(target_aff)
        weights[extreme_mask] = self.extreme_weight
        
        weighted_huber = (huber_base * weights).mean()
        
        # Ranking preservation
        ranking_loss = torch.tensor(0.0, device=pred_aff.device)
        # if self.ranking_weight > 0 and len(pred_aff) > 1:
        #     pred_centered = pred_aff - pred_aff.mean()
        #     target_centered = target_aff - target_aff.mean()
        #     correlation = (pred_centered * target_centered).sum() / (
        #         torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum()) + 1e-8
        #     )
        #     ranking_loss = 1.0 - correlation
        if self.ranking_weight > 0 and pred_aff.numel() > 1:  # Use .numel() instead of len()
            pred_centered = pred_aff - pred_aff.mean()
            target_centered = target_aff - target_aff.mean()
            correlation = (pred_centered * target_centered).sum() / (
                torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum()) + 1e-8
            )
            ranking_loss = 1.0 - correlation 
        total_loss = weighted_huber + self.ranking_weight * ranking_loss
        # Return same format as your original loss for compatibility
        return (total_loss, weighted_huber, ranking_loss, 
                torch.tensor(0.0), torch.tensor(0.0))

class WeightedCategoryLoss_v2(nn.Module):
    def __init__(self,
                 thresholds=None,
                 category_weights=None,
                 regression_weight=0.7,
                 category_penalty_weight=0.2,
                 extreme_penalty_weight=0.1,
                 pearson_penalty_weight=0.0,
                 relative_error_weight=0.0,
                 extreme_boost_low=1.0,
                 extreme_boost_high=1.0):
        super().__init__()

        self.thresholds = torch.tensor(thresholds) if thresholds is not None else torch.tensor(
            [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
        self.category_weights = torch.tensor(category_weights) if category_weights is not None else torch.ones(self.thresholds.size(0) + 1)
        # Normalize weights (optional soft tuning)
        self.category_weights = self.category_weights / self.category_weights.min()

        self.regression_weight = regression_weight
        self.category_penalty_weight = category_penalty_weight
        self.extreme_penalty_weight = extreme_penalty_weight
        self.pearson_penalty_weight = pearson_penalty_weight
        self.relative_error_weight = relative_error_weight

        self.extreme_boost_low = extreme_boost_low
        self.extreme_boost_high = extreme_boost_high

        self.mse_loss = nn.MSELoss(reduction='none')
        self.data_mean = 6.495
        self.data_std = 1.833

    def get_category(self, affinity):
        thresholds = self.thresholds.to(affinity.device)
        return torch.searchsorted(thresholds, affinity, right=False)

    def pearson_corr(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        corr = torch.sum(vx * vy) / (torch.norm(vx) * torch.norm(vy) + 1e-8)
        return corr

    def forward(self, pred_aff, target_aff, pred_logits=None):
        mse_loss = self.mse_loss(pred_aff, target_aff)

        # Category-based weights
        target_categories = self.get_category(target_aff)
        base_weights = self.category_weights.to(pred_aff.device)[target_categories]

        # Extreme boost
        extreme_weights = torch.ones_like(target_aff)
        lower_threshold = self.thresholds[0].item() if self.thresholds.numel() > 0 else 4.0
        upper_threshold = self.thresholds[-1].item() if self.thresholds.numel() > 0 else 9.0
        extreme_weights[target_aff < lower_threshold] = self.extreme_boost_low
        extreme_weights[target_aff > upper_threshold] = self.extreme_boost_high
        total_weights = base_weights * extreme_weights

        weighted_reg_loss = (mse_loss * total_weights).mean()

        # Category distance penalty (ordinal soft penalty)
        pred_categories = self.get_category(pred_aff)
        category_distance = torch.abs(target_categories.float() - pred_categories.float()).mean()

        # Extreme preservation
        extreme_mask = (target_aff < 4.5) | (target_aff > 8.5)
        extreme_penalty = torch.tensor(0.0, device=pred_aff.device)
        if extreme_mask.any():
            pred_distance = torch.abs(pred_aff[extreme_mask] - self.data_mean)
            target_distance = torch.abs(target_aff[extreme_mask] - self.data_mean)
            extreme_penalty = torch.relu(target_distance - pred_distance).mean()

        # Pearson penalty (for ranking signal)
        pearson_penalty = 1.0 - self.pearson_corr(pred_aff, target_aff)

        # Relative error penalty
        rel_error = (torch.abs(pred_aff - target_aff) / (target_aff + 1e-3)).mean()

        total_loss = (
            self.regression_weight * weighted_reg_loss +
            self.category_penalty_weight * category_distance +
            self.extreme_penalty_weight * extreme_penalty +
            self.pearson_penalty_weight * pearson_penalty +
            self.relative_error_weight * rel_error
        )

        return total_loss, weighted_reg_loss, category_distance, extreme_penalty, pearson_penalty


class RankingMSELoss(nn.Module):
    """
    Combines MSE loss with pairwise ranking loss to ensure relative ordering.
    """
    def __init__(self, mse_weight=0.7, ranking_weight=0.3, margin=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.ranking_weight = ranking_weight
        self.margin = margin
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        # Standard MSE loss
        mse_loss = self.mse_loss(pred, target)
        
        # Pairwise ranking loss
        n = pred.size(0)
        if n < 2:
            return mse_loss
        
        ranking_loss = 0.0
        count = 0
        
        # Compare all pairs
        for i in range(n):
            for j in range(i+1, n):
                # If target[i] > target[j], then pred[i] should be > pred[j]
                if target[i] > target[j]:
                    # We want pred[i] - pred[j] > margin
                    loss = F.relu(self.margin - (pred[i] - pred[j]))
                elif target[i] < target[j]:
                    # We want pred[j] - pred[i] > margin
                    loss = F.relu(self.margin - (pred[j] - pred[i]))
                else:
                    # Equal targets, minimize difference
                    loss = torch.abs(pred[i] - pred[j])
                
                ranking_loss += loss
                count += 1
        
        if count > 0:
            ranking_loss = ranking_loss / count
        
        total_loss = self.mse_weight * mse_loss + self.ranking_weight * ranking_loss
        
        return total_loss


class AdaptiveHuberLoss(nn.Module):
    """
    Huber loss with adaptive delta based on target distribution.
    Good for handling outliers while maintaining sensitivity in the common range.
    """
    def __init__(self, delta=1.0, adaptive=True, quantiles=[0.1, 0.9]):
        super().__init__()
        self.delta = delta
        self.adaptive = adaptive
        self.quantiles = quantiles
        
    def forward(self, pred, target):
        # Adaptively set delta based on target distribution
        if self.adaptive and target.numel() > 10:
            # Use the IQR of current batch to set delta
            sorted_targets, _ = torch.sort(target)
            q1_idx = int(self.quantiles[0] * len(sorted_targets))
            q2_idx = int(self.quantiles[1] * len(sorted_targets))
            iqr = sorted_targets[q2_idx] - sorted_targets[q1_idx]
            delta = max(iqr * 0.5, 0.5)  # At least 0.5
        else:
            delta = self.delta
        
        # Compute Huber loss
        residual = torch.abs(pred - target)
        mask = residual <= delta
        
        # L2 loss for small residuals, L1 for large
        loss = torch.where(
            mask,
            0.5 * residual ** 2,
            delta * (residual - 0.5 * delta)
        )
        
        return loss.mean()


class FocalMSELoss(nn.Module):
    """
    MSE loss with focal weighting - focuses on hard examples.
    """
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, pred, target):
        # Compute squared error
        se = (pred - target) ** 2
        
        # Normalize errors to [0, 1] range for focal weighting
        max_error = se.max().detach()
        if max_error > 0:
            normalized_error = se / max_error
        else:
            normalized_error = se
        
        # Focal weight: higher weight for larger errors
        focal_weight = self.alpha * (normalized_error ** self.gamma)
        
        # Weighted loss
        loss = focal_weight * se
        
        return loss.mean()


class MultiTaskAffinityLoss(nn.Module):
    """
    Multi-task loss that treats affinity prediction as both regression and ordinal classification.
    """
    def __init__(self, num_classes=12, class_weight=0.3, reg_weight=0.7):
        super().__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.reg_weight = reg_weight
        
        # Define bins for ordinal classification based on your data range
        self.min_affinity = 2.0
        self.max_affinity = 15.5
        self.bin_width = (self.max_affinity - self.min_affinity) / num_classes
        
        self.mse_loss = nn.MSELoss()
        
    def affinity_to_class(self, affinity):
        """Convert continuous affinity to class index"""
        # Clip values to valid range
        affinity = torch.clamp(affinity, self.min_affinity, self.max_affinity)
        
        # Convert to class
        class_idx = ((affinity - self.min_affinity) / self.bin_width).long()
        class_idx = torch.clamp(class_idx, 0, self.num_classes - 1)
        
        return class_idx
    
    def forward(self, pred_affinity, pred_logits, target_affinity):
        """
        Args:
            pred_affinity: predicted continuous affinity values
            pred_logits: predicted class logits (from auxiliary head)
            target_affinity: ground truth affinity values
        """
        # Regression loss
        reg_loss = self.mse_loss(pred_affinity, target_affinity)
        
        # Classification loss
        target_classes = self.affinity_to_class(target_affinity)
        
        # Use ordinal cross-entropy (considers ordering)
        # Create soft labels that acknowledge nearby classes
        soft_labels = torch.zeros(target_classes.size(0), self.num_classes, device=target_classes.device)
        
        for i in range(len(target_classes)):
            class_idx = target_classes[i]
            soft_labels[i, class_idx] = 0.7  # Main class
            
            # Adjacent classes get some weight
            if class_idx > 0:
                soft_labels[i, class_idx - 1] = 0.15
            if class_idx < self.num_classes - 1:
                soft_labels[i, class_idx + 1] = 0.15
        
        # Ensure probabilities sum to 1
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        
        # Cross entropy with soft labels
        log_probs = F.log_softmax(pred_logits, dim=1)
        class_loss = -(soft_labels * log_probs).sum(dim=1).mean()
        
        # Combined loss
        total_loss = self.reg_weight * reg_loss + self.class_weight * class_loss
        
        return total_loss, reg_loss, class_loss


# Utility functions for training
def compute_affinity_statistics(dataloader):
    """
    Compute statistics of affinity distribution for better loss design
    """
    all_affinities = []
    
    for batch in dataloader:
        affinities = batch['affinity'].cpu().numpy()
        all_affinities.extend(affinities)
    
    all_affinities = np.array(all_affinities)
    
    stats = {
        'mean': np.mean(all_affinities),
        'std': np.std(all_affinities),
        'min': np.min(all_affinities),
        'max': np.max(all_affinities),
        'quantiles': np.percentile(all_affinities, [0, 10, 25, 50, 75, 90, 100]),
        'histogram': np.histogram(all_affinities, bins=20)
    }
    
    # Compute category frequencies for weighted loss
    thresholds = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
    category_counts = []
    
    for i in range(len(thresholds) + 1):
        if i == 0:
            count = np.sum(all_affinities < thresholds[0])
        elif i == len(thresholds):
            count = np.sum(all_affinities >= thresholds[-1])
        else:
            count = np.sum((all_affinities >= thresholds[i-1]) & (all_affinities < thresholds[i]))
        category_counts.append(count)
    
    # Compute inverse frequency weights
    total_samples = len(all_affinities)
    category_weights = [total_samples / (count + 1) for count in category_counts]
    
    # Normalize weights
    min_weight = min(category_weights)
    category_weights = [w / min_weight for w in category_weights]  # Normalize to min=1.0
    
    stats['category_counts'] = category_counts
    stats['category_weights'] = category_weights
    
    return stats


def get_balanced_sampler(dataset, batch_size=32):
    """
    Get a balanced sampler for the dataset based on affinity distribution
    """
    # Extract affinities
    affinities = []
    for i in range(len(dataset)):
        affinities.append(dataset[i]['affinity'].item())
    
    affinities = np.array(affinities)
    thresholds = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
    
    # Assign categories
    categories = np.searchsorted(thresholds, affinities)
    
    # Compute weights for each sample
    category_counts = np.bincount(categories)
    weights = 1.0 / category_counts[categories]
    weights = weights / weights.sum()
    
    # Create sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(dataset),
        replacement=True
    )
    
    return sampler