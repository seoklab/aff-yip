import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class WeightedCategoryLoss(nn.Module):
    """
    Combines regression loss with category-based weighting to handle imbalanced affinity distribution.
    Enhanced version based on actual data statistics.
    """
    def __init__(self, thresholds=None, category_weights=None, regression_weight=0.7, 
                 extreme_boost_low=2.0, extreme_boost_high=1.5, 
                 category_penalty_weight=0.2, extreme_penalty_weight=0.1):
        super().__init__()
        if thresholds is None:
            # Default thresholds for affinity categories
            self.thresholds = torch.tensor([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
        else:
            self.thresholds = torch.tensor(thresholds)
        
        if category_weights is None:
            # Using your actual computed weights
            self.category_weights = torch.tensor([10.38, 17.96, 15.04, 12.26, 12.70, 9.99, 10.47, 8.92, 9.95, 12.35, 19.48, 12.41])
        else:
            self.category_weights = torch.tensor(category_weights)
        
        # Normalize weights to prevent exploding gradients
        self.category_weights = self.category_weights / self.category_weights.min()
        
        self.regression_weight = regression_weight
        self.category_penalty_weight = category_penalty_weight
        self.extreme_penalty_weight = extreme_penalty_weight
        self.extreme_boost_low = extreme_boost_low
        self.extreme_boost_high = extreme_boost_high
        
        # self.mse_loss = nn.MSELoss(reduction='none')
        self.mse_loss = nn.HuberLoss(reduction='none', delta=1.0)

        # Data statistics for extreme penalty
        self.data_mean = 6.495
        self.data_std = 1.833
        
    def get_category(self, affinity):
        """Get category index for each affinity value using searchsorted for efficiency"""
        thresholds = self.thresholds.to(affinity.device)
        # Use searchsorted for vectorized category assignment
        categories = torch.searchsorted(thresholds, affinity, right=False)
        return categories
    def forward(self, pred_aff, target_aff, pred_logits=None):
        """
        pred_aff: [B] - predicted affinity (regression)
        target_aff: [B] - ground truth affinity
        pred_logits: [B, 12] - predicted logits for affinity category (optional)
        """
        mse_loss = self.mse_loss(pred_aff, target_aff)
        target_categories = self.get_category(target_aff)  # [B]
        base_weights = self.category_weights.to(pred_aff.device)[target_categories]

        # Extreme boost
        extreme_weights = torch.ones_like(target_aff)
        extreme_weights[target_aff < 4.0] = self.extreme_boost_low
        extreme_weights[target_aff > 9.0] = self.extreme_boost_high
        total_weights = base_weights * extreme_weights

        weighted_reg_loss = (mse_loss * total_weights).mean()

        # Category penalty - soft difference in predicted category vs true category
        pred_categories = self.get_category(pred_aff)
        category_distance = torch.abs(target_categories.float() - pred_categories.float()).mean()

        # Extreme preservation penalty
        extreme_mask = (target_aff < 4.5) | (target_aff > 8.5)
        extreme_penalty = torch.tensor(0.0, device=pred_aff.device)
        if extreme_mask.any():
            pred_distance = torch.abs(pred_aff[extreme_mask] - self.data_mean)
            target_distance = torch.abs(target_aff[extreme_mask] - self.data_mean)
            extreme_penalty = torch.relu(target_distance - pred_distance).mean()

        total_loss = (
            self.regression_weight * weighted_reg_loss +
            self.category_penalty_weight * category_distance +
            self.extreme_penalty_weight * extreme_penalty
        )

        # === Classification Loss ===
        cls_loss = torch.tensor(0.0, device=pred_aff.device)
        if pred_logits is not None:
            cls_loss = F.cross_entropy(pred_logits, target_categories, weight=self.category_weights.to(pred_logits.device), label_smoothing=0.1)
        #     total_loss += 0.2 * cls_loss  # optional scaling factor for classification

        return total_loss, weighted_reg_loss, category_distance, extreme_penalty, cls_loss

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