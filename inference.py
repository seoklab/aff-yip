import os
import argparse
import torch
import pytorch_lightning as pl
import json
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Union

from src.model.basic import AFFModel_GVP
from src.model.basic_three import AFFModel_ThreeBody 
from src.model.multi_three import AFFModel_ThreeBody as AFFModel_MultiThreeBody
from src.model.multi_two import AFFModel_TwoBody
from src.data.datamodule import RLADataModule


def load_model_from_checkpoint(checkpoint_path: str, model_type: str, **model_kwargs):
    """Load model from checkpoint based on model type"""
    
    if model_type == 'three_body':
        model = AFFModel_ThreeBody.load_from_checkpoint(checkpoint_path, **model_kwargs)
    elif model_type == 'multi':
        model = AFFModel_MultiThreeBody.load_from_checkpoint(checkpoint_path, **model_kwargs)
    elif model_type == 'two_body':
        model = AFFModel_TwoBody.load_from_checkpoint(checkpoint_path, **model_kwargs)
    else:  # default to gvp
        model = AFFModel_GVP.load_from_checkpoint(checkpoint_path, **model_kwargs)
    
    model.eval()
    return model


def run_inference_on_dataloader(model, dataloader, device):
    """Run inference on a dataloader and return predictions and targets"""
    predictions = []
    targets = []
    classification_logits = []
    
    model.to(device)
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                batch = batch.to(device)
            
            # Get model predictions - handle your specific model output format
            output = model(batch)
            
            # Handle your model's specific output format
            if hasattr(model, 'loss_type') and model.loss_type == "multitask":
                # Multitask model returns: pred_affinity, pred_logits, affinities
                if isinstance(output, tuple) and len(output) == 3:
                    pred_affinity, pred_logits, affinities = output
                    predictions.append(pred_affinity.cpu().numpy())
                    targets.append(affinities.cpu().numpy())
                    if pred_logits is not None:
                        classification_logits.append(pred_logits.cpu().numpy())
                else:
                    # Fallback - try to extract affinity prediction
                    pred_affinity = output[0] if isinstance(output, tuple) else output
                    predictions.append(pred_affinity.cpu().numpy())
                    
            elif hasattr(model, 'loss_type') and model.loss_type == "single":
                # Single task model returns: pred_affinity, affinities
                if isinstance(output, tuple) and len(output) == 2:
                    pred_affinity, affinities = output
                    predictions.append(pred_affinity.cpu().numpy())
                    targets.append(affinities.cpu().numpy())
                else:
                    pred_affinity = output[0] if isinstance(output, tuple) else output
                    predictions.append(pred_affinity.cpu().numpy())
                    
            else:
                # Generic handling for other model types
                if isinstance(output, tuple):
                    pred = output[0]  # Assume first element is the main prediction
                elif isinstance(output, dict):
                    pred = output.get('predictions', output.get('pred', output.get('logits')))
                else:
                    pred = output
                
                predictions.append(pred.cpu().numpy())
            
            # Try to get targets from batch if not already extracted
            if len(targets) <= len(predictions) - 1:
                if 'affinities' in batch:
                    targets.append(batch['affinities'].cpu().numpy())
                elif 'target' in batch:
                    targets.append(batch['target'].cpu().numpy())
                elif 'y' in batch:
                    targets.append(batch['y'].cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0) if targets else None
    classification_logits = np.concatenate(classification_logits, axis=0) if classification_logits else None
    
    return predictions, targets, classification_logits


def run_inference_single_sample(model, data_module, sample_data, device):
    """Run inference on a single sample"""
    model.to(device)
    model.eval()
    
    # Process single sample through data module if needed
    # This depends on your data format - you may need to adapt this
    with torch.no_grad():
        if isinstance(sample_data, dict):
            sample_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in sample_data.items()}
        else:
            sample_data = sample_data.to(device)
        
        output = model(sample_data)
        
        # Handle your model's specific output format
        if hasattr(model, 'loss_type') and model.loss_type == "multitask":
            if isinstance(output, tuple) and len(output) == 3:
                pred_affinity, pred_logits, _ = output
                return pred_affinity.cpu().numpy(), pred_logits.cpu().numpy() if pred_logits is not None else None
        elif hasattr(model, 'loss_type') and model.loss_type == "single":
            if isinstance(output, tuple) and len(output) == 2:
                pred_affinity, _ = output
                return pred_affinity.cpu().numpy(), None
        
        # Fallback
        pred = output[0] if isinstance(output, tuple) else output
        return pred.cpu().numpy(), None


def calculate_metrics(predictions, targets):
    """Calculate common regression metrics"""
    if targets is None:
        return {}
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import pearsonr, spearmanr
    
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    # Correlation coefficients
    pearson_corr, pearson_p = pearsonr(targets.flatten(), predictions.flatten())
    spearman_corr, spearman_p = spearmanr(targets.flatten(), predictions.flatten())
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_corr': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p_value': spearman_p
    }
    
    return metrics


def save_results(predictions, targets, output_path, metrics=None, sample_ids=None, classification_logits=None):
    """Save inference results to files"""
    results = {
        'predictions': predictions.tolist(),
    }
    
    if targets is not None:
        results['targets'] = targets.tolist()
    
    if classification_logits is not None:
        results['classification_logits'] = classification_logits.tolist()
    
    if sample_ids is not None:
        results['sample_ids'] = sample_ids
    
    if metrics is not None:
        results['metrics'] = metrics
    
    # Save as JSON
    with open(output_path.replace('.json', '_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as CSV for easy viewing
    df_data = {'predictions': predictions.flatten()}
    if targets is not None:
        df_data['targets'] = targets.flatten()
        df_data['residuals'] = targets.flatten() - predictions.flatten()
        df_data['abs_residuals'] = np.abs(targets.flatten() - predictions.flatten())
    
    if classification_logits is not None:
        # If classification logits are present, add class predictions
        if len(classification_logits.shape) > 1:
            df_data['predicted_class'] = np.argmax(classification_logits, axis=1)
            # Add class probabilities if it's a softmax output
            for i in range(classification_logits.shape[1]):
                df_data[f'class_{i}_prob'] = classification_logits[:, i]
        else:
            df_data['classification_logits'] = classification_logits.flatten()
    
    if sample_ids is not None:
        df_data['sample_id'] = sample_ids
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_path.replace('.json', '_results.csv'), index=False)
    
    # Save metrics separately
    if metrics:
        with open(output_path.replace('.json', '_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model from checkpoint
    print(f"Loading model from: {args.checkpoint_path}")
    model = load_model_from_checkpoint(
        args.checkpoint_path, 
        args.model_type,
        strict=False  # Allow loading with some missing keys if needed
    )
    
    print(f"Loaded {args.model_type} model successfully")
    
    # Setup data module
    data_module = RLADataModule(
        data_path=args.data_path,
        train_meta_path=args.train_data_path,
        val_meta_path=args.valid_data_path,
        test_meta_path=args.test_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        top_k=args.top_k,
        crop_size=args.crop_size
    )
    
    # Setup data module
    data_module.setup('test')
    
    # Run inference based on mode
    if args.mode == 'test':
        print("Running inference on test set...")
        test_loader = data_module.test_dataloader()
        predictions, targets, classification_logits = run_inference_on_dataloader(model, test_loader, device)
        
    elif args.mode == 'val':
        print("Running inference on validation set...")
        val_loader = data_module.val_dataloader()
        predictions, targets, classification_logits = run_inference_on_dataloader(model, val_loader, device)
        
    elif args.mode == 'train':
        print("Running inference on training set...")
        train_loader = data_module.train_dataloader()
        predictions, targets, classification_logits = run_inference_on_dataloader(model, train_loader, device)
        
    else:  # custom data
        if not args.custom_data_path:
            raise ValueError("Must provide --custom_data_path when using custom mode")
        
        print(f"Running inference on custom data: {args.custom_data_path}")
        # You'll need to implement loading custom data based on your format
        # This is a placeholder - adapt based on your data structure
        raise NotImplementedError("Custom data inference not implemented - please adapt based on your data format")
    
    # Print basic info about results
    print(f"\nInference completed:")
    print(f"  Predictions shape: {predictions.shape}")
    if targets is not None:
        print(f"  Targets shape: {targets.shape}")
    if classification_logits is not None:
        print(f"  Classification logits shape: {classification_logits.shape}")
    
    # Check for multitask model output
    if hasattr(model, 'loss_type'):
        print(f"  Model loss type: {model.loss_type}")
        if model.loss_type == "multitask" and classification_logits is not None:
            print(f"  Multitask model detected - saved both affinity predictions and classification logits")
    
    # Calculate metrics if targets are available
    metrics = None
    if targets is not None:
        print("Calculating metrics...")
        metrics = calculate_metrics(predictions, targets)
        
        print("\nResults:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Save results
    if args.output_path:
        print(f"Saving results to: {args.output_path}")
        save_results(predictions, targets, args.output_path, metrics, classification_logits=classification_logits)
        print("Results saved successfully!")
    
    return predictions, targets, metrics, classification_logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with trained protein-ligand affinity models')
    
    # Model and checkpoint
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the model checkpoint file')
    parser.add_argument('--model_type', type=str, choices=['gvp', 'three_body', 'multi', 'two_body'], 
                       default='gvp', help='Model type (must match the checkpoint)')
    
    # Data paths (reuse from training script)
    parser.add_argument('--data_path', type=str, default='.')
    parser.add_argument('--train_data_path', type=str, default='data/train_1.json')
    parser.add_argument('--valid_data_path', type=str, default='data/val.json')
    parser.add_argument('--test_data_path', type=str, default='data/casp.json')
    parser.add_argument('--custom_data_path', type=str, help='Path to custom data for inference')
    
    # Data processing
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--crop_size', type=int, default=30)
    
    # Inference settings
    parser.add_argument('--mode', type=str, choices=['test', 'val', 'train', 'custom'], 
                       default='test', help='Which dataset to run inference on')
    parser.add_argument('--output_path', type=str, default='inference_results.json',
                       help='Path to save inference results')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    # Create output directory if needed
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    
    # Run inference
    predictions, targets, metrics, classification_logits = main(args)