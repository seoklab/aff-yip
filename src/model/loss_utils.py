import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AffHuberLoss(nn.Module):
    def __init__(self,
                 beta=1.0,                    # Huber threshold
                 extreme_weight=2.0,           # Boost for extremes
                 ranking_weight=0.1,           # Preserve correlations
                 **kwargs):                    # Absorb unused parameters
        super().__init__()
        self.beta = beta
        self.extreme_weight = extreme_weight
        self.ranking_weight = ranking_weight

    def forward(self, pred_aff, target_aff, pred_logits=None):
        # Main Huber loss
        huber_base = F.smooth_l1_loss(pred_aff, target_aff, beta=self.beta, reduction='none')

        # Extreme value emphasis
        extreme_mask = (target_aff < 4.5) | (target_aff > 8.5)
        weights = torch.ones_like(target_aff)
        weights[extreme_mask] = self.extreme_weight

        weighted_huber = (huber_base * weights).mean()

        # Ranking preservation
        ranking_loss = torch.tensor(0.0, device=pred_aff.device)
        if self.ranking_weight > 0 and pred_aff.numel() > 1:  # Use .numel() instead of len()
            pred_centered = pred_aff - pred_aff.mean()
            target_centered = target_aff - target_aff.mean()
            correlation = (pred_centered * target_centered).sum() / (
                torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum()) + 1e-8
            )
            ranking_loss = 1.0 - correlation
        total_loss = weighted_huber + self.ranking_weight * ranking_loss
        
        return (total_loss, weighted_huber, ranking_loss,
                torch.tensor(0.0), torch.tensor(0.0))
    
# Modified Loss Function for Sidechain Map Approach
class SidechainMapCoordinateLoss(nn.Module):
    """
    Loss function that works directly with sidechain_map format.
    Now includes intra-molecular distance preservation loss for geometric consistency.
    """

    def __init__(self,
                 loss_type="mse",
                 ligand_weight=1.0,
                 sidechain_weight=0.5,
                 use_distance_loss=True,
                 ligand_distance_weight=0.2,
                 sidechain_distance_weight=0.1,
                 distance_loss_type="mse",
                 distance_cutoff=5.0):
        super().__init__()
        self.loss_type = loss_type
        self.ligand_weight = ligand_weight
        self.sidechain_weight = sidechain_weight
        self.use_distance_loss = use_distance_loss
        self.ligand_distance_weight = ligand_distance_weight
        self.sidechain_distance_weight = sidechain_distance_weight
        self.distance_loss_type = distance_loss_type
        self.distance_cutoff = distance_cutoff  # Only consider distances below this threshold

    def compute_distance_matrix(self, coords):
        """
        Compute pairwise distance matrix for a set of coordinates.
        Args:
            coords: tensor of shape [N, 3]
        Returns:
            distance_matrix: tensor of shape [N, N]
        """
        # coords: [N, 3]
        # Compute pairwise distances efficiently
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N, N, 3]
        distances = torch.norm(diff, dim=2)  # [N, N]
        return distances

    def distance_loss_fn(self, pred_distances, target_distances, mask=None):
        """
        Compute distance preservation loss between predicted and target distance matrices.
        """
        if mask is not None:
            pred_distances = pred_distances[mask]
            target_distances = target_distances[mask]

        if self.distance_loss_type == "mse":
            return F.mse_loss(pred_distances, target_distances)
        elif self.distance_loss_type == "huber":
            return F.huber_loss(pred_distances, target_distances, delta=1.0)
        elif self.distance_loss_type == "l1":
            return F.l1_loss(pred_distances, target_distances)
        else:
            return F.mse_loss(pred_distances, target_distances)

    def compute_ligand_distance_loss(self, predictions):
        """
        Compute distance preservation loss for ligand molecules.
        """
        distance_losses = []

        for batch_id, batch_data in predictions['ligand_coords'].items():
            if isinstance(batch_data, dict) and 'predictions' in batch_data and 'targets' in batch_data:
                pred_coords = batch_data['predictions']
                target_coords = batch_data['targets']

                if pred_coords.numel() > 0 and target_coords.numel() > 0 and pred_coords.size(0) > 1:
                    # Compute distance matrices
                    pred_distances = self.compute_distance_matrix(pred_coords)
                    target_distances = self.compute_distance_matrix(target_coords)

                    # Create mask to only consider distances below cutoff
                    # Also exclude diagonal (self-distances = 0)
                    mask = (target_distances > 0) & (target_distances <= self.distance_cutoff)

                    if mask.sum() > 0:  # Only compute loss if we have valid distances
                        batch_distance_loss = self.distance_loss_fn(pred_distances, target_distances, mask)
                        distance_losses.append(batch_distance_loss)

        if distance_losses:
            return torch.stack(distance_losses).mean()
        else:
            return torch.tensor(0.0, device=pred_coords.device if 'pred_coords' in locals() else 'cpu')

    def compute_sidechain_distance_loss(self, predictions):
        """
        Compute distance preservation loss for sidechain atoms within each residue.
        """
        distance_losses = []

        for batch_id in predictions['sidechain_predictions']:
            if batch_id not in predictions['sidechain_targets']:
                continue

            batch_predictions = predictions['sidechain_predictions'][batch_id]
            batch_targets = predictions['sidechain_targets'][batch_id]

            for residue_key in batch_predictions:
                if residue_key not in batch_targets:
                    continue

                residue_predictions = batch_predictions[residue_key]
                residue_targets = batch_targets[residue_key]

                # Get common atoms that exist in both prediction and target
                common_atoms = set(residue_predictions.keys()) & set(residue_targets.keys())

                if len(common_atoms) > 1:  # Need at least 2 atoms for distances
                    # Collect coordinates for this residue
                    pred_coords_list = []
                    target_coords_list = []

                    for atom_name in sorted(common_atoms):  # Sort for consistent ordering
                        pred_coords_list.append(residue_predictions[atom_name])
                        target_coords_list.append(residue_targets[atom_name])

                    # Stack into tensors
                    pred_coords = torch.stack(pred_coords_list)  # [N_atoms, 3]
                    target_coords = torch.stack(target_coords_list)  # [N_atoms, 3]

                    # Compute distance matrices
                    pred_distances = self.compute_distance_matrix(pred_coords)
                    target_distances = self.compute_distance_matrix(target_coords)

                    # Create mask to exclude diagonal and only consider reasonable distances
                    mask = (target_distances > 0) & (target_distances <= self.distance_cutoff)

                    if mask.sum() > 0:
                        residue_distance_loss = self.distance_loss_fn(pred_distances, target_distances, mask)
                        distance_losses.append(residue_distance_loss)

        if distance_losses:
            return torch.stack(distance_losses).mean()
        else:
            # Return zero loss on appropriate device
            if predictions['sidechain_predictions']:
                first_batch = list(predictions['sidechain_predictions'].values())[0]
                if first_batch:
                    first_residue = list(first_batch.values())[0]
                    if first_residue:
                        first_atom = list(first_residue.values())[0]
                        return torch.tensor(0.0, device=first_atom.device)
            return torch.tensor(0.0)

    def forward(self, predictions, targets=None):
        """
        Args:
            predictions: Dict from EGNN model with structure:
                {
                    'ligand_coords': {batch_id: {'predictions': tensor, 'targets': tensor}},
                    'sidechain_predictions': {batch_id: {residue_key: {atom_name: tensor[3]}}},
                    'sidechain_targets': {batch_id: {residue_key: {atom_name: tensor[3]}}}
                }
        """
        total_loss = 0.0
        loss_dict = {}
        total_atoms = 0

        # Standard coordinate loss (ligand)
        if 'ligand_coords' in predictions and predictions['ligand_coords']:
            ligand_losses = []
            ligand_atoms = 0

            for batch_id, batch_data in predictions['ligand_coords'].items():
                if isinstance(batch_data, dict) and 'predictions' in batch_data:
                    pred_coords = batch_data['predictions']
                    target_coords = batch_data['targets']

                    if pred_coords.numel() > 0 and target_coords.numel() > 0:
                        if self.loss_type == "mse":
                            batch_loss = F.mse_loss(pred_coords, target_coords)
                        elif self.loss_type == "huber":
                            batch_loss = F.huber_loss(pred_coords, target_coords, delta=1.0)
                        else:
                            batch_loss = F.mse_loss(pred_coords, target_coords)

                        ligand_losses.append(batch_loss)
                        ligand_atoms += pred_coords.size(0)

            if ligand_losses:
                ligand_loss = torch.stack(ligand_losses).mean()
                total_loss += self.ligand_weight * ligand_loss
                loss_dict['ligand_coord_loss'] = ligand_loss.item()

                # Calculate RMSD for logging
                ligand_rmsd = ligand_loss.sqrt()
                loss_dict['ligand_rmsd'] = ligand_rmsd.item()

                # Add ligand distance loss
                if self.use_distance_loss:
                    ligand_distance_loss = self.compute_ligand_distance_loss(predictions)
                    total_loss += self.ligand_distance_weight * ligand_distance_loss
                    loss_dict['ligand_distance_loss'] = ligand_distance_loss.item()

        # Standard coordinate loss (sidechain)
        if ('sidechain_predictions' in predictions and
            'sidechain_targets' in predictions):

            sidechain_losses = []
            sidechain_atoms = 0

            for batch_id in predictions['sidechain_predictions']:
                if batch_id not in predictions['sidechain_targets']:
                    continue

                batch_predictions = predictions['sidechain_predictions'][batch_id]
                batch_targets = predictions['sidechain_targets'][batch_id]

                for residue_key in batch_predictions:
                    if residue_key not in batch_targets:
                        continue

                    residue_predictions = batch_predictions[residue_key]
                    residue_targets = batch_targets[residue_key]

                    # Only compute loss for atoms that exist in both prediction and target
                    common_atoms = set(residue_predictions.keys()) & set(residue_targets.keys())

                    for atom_name in common_atoms:
                        pred_coord = residue_predictions[atom_name]
                        target_coord = residue_targets[atom_name]

                        if self.loss_type == "mse":
                            atom_loss = F.mse_loss(pred_coord, target_coord)
                        elif self.loss_type == "huber":
                            atom_loss = F.huber_loss(pred_coord, target_coord, delta=1.0)
                        else:
                            atom_loss = F.mse_loss(pred_coord, target_coord)

                        sidechain_losses.append(atom_loss)
                        sidechain_atoms += 1

            if sidechain_losses:
                sidechain_loss = torch.stack(sidechain_losses).mean()
                total_loss += self.sidechain_weight * sidechain_loss
                loss_dict['sidechain_coord_loss'] = sidechain_loss.item()

                # Calculate RMSD for logging
                rmsd_sum = 0
                for loss_val in sidechain_losses:
                    rmsd_sum += loss_val.sqrt()
                sidechain_rmsd = rmsd_sum / len(sidechain_losses)
                loss_dict['sidechain_rmsd'] = sidechain_rmsd.item()

                # Add sidechain distance loss
                if self.use_distance_loss:
                    sidechain_distance_loss = self.compute_sidechain_distance_loss(predictions)
                    total_loss += self.sidechain_distance_weight * sidechain_distance_loss
                    loss_dict['sidechain_distance_loss'] = sidechain_distance_loss.item()

            total_atoms += sidechain_atoms

        loss_dict['total_coord_loss'] = total_loss.item() if torch.is_tensor(total_loss) else total_loss
        loss_dict['total_atoms_predicted'] = total_atoms

        return total_loss, loss_dict


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



### unused loss functions ###

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


class CoordinateLoss(nn.Module):
    """
    Simple coordinate-based loss functions for structure prediction.
    Calculates RMSD and distance-based losses between predicted and target coordinates.
    """

    def __init__(self,
                 loss_type="mse",  # "mse", "rmsd", "huber"
                 ligand_weight=1.0,
                 sidechain_weight=0.5,
                 delta=1.0):  # For Huber loss
        super().__init__()
        self.loss_type = loss_type
        self.ligand_weight = ligand_weight
        self.sidechain_weight = sidechain_weight
        self.delta = delta

    def forward(self,
                pred_ligand_coords,       # [B, max_lig_atoms, 3]
                target_ligand_coords,     # [B, max_lig_atoms, 3]
                lig_coord_mask,              # [B, max_lig_atoms]
                pred_sidechain_coords=None,    # [B, N_residues, max_sidechain_atoms, 3]
                target_sidechain_coords=None,  # [B, N_residues, max_sidechain_atoms, 3]
                sidechain_mask=None,          # [B, N_residues, max_sidechain_atoms]
                prot_coord_mask=None):

        total_loss = 0.0
        loss_dict = {}

        # ===== LIGAND COORDINATE LOSS =====
        if pred_ligand_coords is not None and target_ligand_coords is not None:
            ligand_loss = self._compute_coordinate_loss(
                pred_ligand_coords,
                target_ligand_coords,
                lig_coord_mask,
            )
            total_loss += self.ligand_weight * ligand_loss
            loss_dict['ligand_coord_loss'] = ligand_loss.item()

            # Calculate RMSD for logging
            ligand_rmsd = self._compute_rmsd(
                pred_ligand_coords,
                target_ligand_coords,
                lig_coord_mask
            )
            loss_dict['ligand_rmsd'] = ligand_rmsd.item()

        # ===== SIDECHAIN COORDINATE LOSS =====
        if (pred_sidechain_coords is not None and
            target_sidechain_coords is not None and
            sidechain_mask is not None):
            combined_mask = prot_coord_mask.unsqueeze(-1) & sidechain_mask  # [B, N_residues, max_sidechain_atoms]

            sidechain_loss = self._compute_coordinate_loss(
                pred_sidechain_coords.view(-1, 3),
                target_sidechain_coords.view(-1, 3),
                combined_mask.view(-1)
            )

            total_loss += self.sidechain_weight * sidechain_loss
            loss_dict['sidechain_coord_loss'] = sidechain_loss.item()

            # Calculate RMSD for logging
            sidechain_rmsd = self._compute_rmsd(
                pred_sidechain_coords.view(-1, 3),
                target_sidechain_coords.view(-1, 3),
                combined_mask.view(-1)
            )
            loss_dict['sidechain_rmsd'] = sidechain_rmsd.item()

        loss_dict['total_coord_loss'] = total_loss.item() if torch.is_tensor(total_loss) else total_loss

        return total_loss, loss_dict

    def _compute_coordinate_loss(self, pred_coords, target_coords, mask):
        """
        Compute coordinate loss with masking.

        Args:
            pred_coords: [N, 3] predicted coordinates
            target_coords: [N, 3] target coordinates
            mask: [N] boolean mask for valid coordinates
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_coords.device, requires_grad=True)

        # Apply mask
        pred_masked = pred_coords[mask]
        target_masked = target_coords[mask]

        if self.loss_type == "mse":
            loss = F.mse_loss(pred_masked, target_masked)
        elif self.loss_type == "rmsd":
            loss = self._compute_rmsd_raw(pred_masked, target_masked)
        elif self.loss_type == "huber":
            loss = F.huber_loss(pred_masked, target_masked, delta=self.delta)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    def _compute_rmsd(self, pred_coords, target_coords, mask):
        """Compute RMSD between predicted and target coordinates."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_coords.device)

        pred_masked = pred_coords[mask]
        target_masked = target_coords[mask]

        return self._compute_rmsd_raw(pred_masked, target_masked)

    def _compute_rmsd_raw(self, pred_coords, target_coords):
        """Raw RMSD computation without masking."""
        squared_diff = (pred_coords - target_coords).pow(2).sum(dim=-1)
        rmsd = squared_diff.mean().sqrt()
        return rmsd

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
        self.coord_loss_fn = CoordinateLoss(loss_type=coord_loss_type, ligand_weight=0.75, sidechain_weight=0.25)
        if lig_internal_dist_w > 0:
            self.internal_prediction = True
            self.coord_loss = DistanceBasedCoordinateLoss(coord_weight=coord_loss_weight,
                                                           distance_weight=lig_internal_dist_w,
                                                           coord_loss_type=coord_loss_type)
        else:
            self.internal_prediction = False
            self.coord_loss = CoordinateLoss(loss_type=coord_loss_type)

    def forward(self, pred_affinity, target_affinity, pred_logits=None,
                pred_lig_coords=None, target_lig_coords=None, lig_coord_mask=None,
                pred_sidechain_coords=None, target_sidechain_coords=None, sidechain_mask=None, prot_coord_mask=None):

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

        if pred_lig_coords is not None and target_lig_coords is not None and lig_coord_mask is not None:
            coord_loss, coord_loss_dict = self.coord_loss_fn(
                pred_ligand_coords=pred_lig_coords,
                target_ligand_coords=target_lig_coords,
                lig_coord_mask=lig_coord_mask,
                pred_sidechain_coords=pred_sidechain_coords,
                target_sidechain_coords=target_sidechain_coords,
                sidechain_mask=sidechain_mask,
                prot_coord_mask=prot_coord_mask
            )

            weighted_coord_loss = self.coord_loss_weight * coord_loss
            total_loss += weighted_coord_loss

            # Add coordinate losses to loss dict
            loss_dict.update(coord_loss_dict)
            loss_dict['weighted_coord_loss'] = weighted_coord_loss.item() if torch.is_tensor(weighted_coord_loss) else weighted_coord_loss

        loss_dict['total_loss'] = total_loss.item() if torch.is_tensor(total_loss) else total_loss

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
