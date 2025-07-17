"""
Inference script using YAML configuration system.
"""

import os
import argparse
import torch
import pytorch_lightning as pl
import json
import numpy as np
from pathlib import Path

# Import model and data module
from src.model.model_threebody_virtual import AFFModel_ThreeBody as AFFModel_VirtualWater
from src.model.threebody_virtual_glem import AFFModel_ThreeBody as AFFModel_VirtualWater_GLEM
from src.data.datamodule import RLADataModule
from src.data.dataset_gvp import RLADataset

# Import config utilities
from scripts.utils.config_parser import ConfigParser, create_parser_with_config_support
from scripts.utils.model_config import ModelConfig


def load_model_from_checkpoint(checkpoint_path: str, model_config: ModelConfig):
    """Load model from checkpoint with config."""
    model_kwargs = model_config.to_model_kwargs()
    
    # Check model type and return appropriate model
    if model_config.model_type == "virtual_glem":
        model = AFFModel_VirtualWater_GLEM.load_from_checkpoint(
            checkpoint_path, 
            **model_kwargs
        )
    elif model_config.model_type == "virtual":
        # Default to virtual water model for "virtual" or other types
        model = AFFModel_VirtualWater.load_from_checkpoint(
            checkpoint_path, 
            **model_kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_config.model_type}. Supported types are 'virtual' and 'virtual_glem'.")
    
    model.eval()
    return model


def create_inference_dataloader_from_config(config_parser: ConfigParser):
    """Create dataloader for inference from config."""
    inference_params = config_parser.get_inference_params()
    data_params = config_parser.get_data_params()
    
    # Create a data module like in training, but only setup test
    data_module = RLADataModule(
        data_path=data_params['data_path'],
        test_meta_path=inference_params['meta_path'],
        batch_size=inference_params.get('batch_size', 1),
        num_workers=inference_params.get('num_workers', 4),
        top_k=data_params.get('top_k', 30),
        crop_size=data_params.get('crop_size', 30)
    )
    
    # Setup the data module
    data_module.setup('test')
    
    # Get the test dataloader
    return data_module.test_dataloader()


def run_inference(model, dataloader, device: str = 'cuda'):
    """Run inference on the dataloader."""
    model.to(device)
    model.eval()
    
    predictions = []
    true_values = []
    sample_names = []
    structure_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            else:
                # Handle dict-like batch
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Extract predictions and true values
            if isinstance(outputs, dict):
                pred_affinity = outputs.get('affinity', outputs.get('pred_affinity'))
                pred_structure = outputs.get('structure', outputs.get('pred_structure'))
                
                if 'affinity' in batch:
                    true_affinity = batch['affinity']
                elif 'target' in batch:
                    true_affinity = batch['target']
                elif hasattr(batch, 'y'):
                    true_affinity = batch.y
                else:
                    true_affinity = None
            else:
                pred_affinity = outputs
                pred_structure = None
                if 'affinity' in batch:
                    true_affinity = batch['affinity']
                elif hasattr(batch, 'y'):
                    true_affinity = batch.y
                else:
                    true_affinity = None
            
            # Store results
            if pred_affinity is not None:
                pred_np = pred_affinity.cpu().numpy()
                if pred_np.ndim == 0:  # scalar
                    predictions.append(pred_np.item())
                else:
                    predictions.extend(pred_np.tolist())
            if true_affinity is not None:
                true_np = true_affinity.cpu().numpy()
                if true_np.ndim == 0:  # scalar
                    true_values.append(true_np.item())
                else:
                    true_values.extend(true_np.tolist())
            if pred_structure is not None:
                structure_predictions.extend(pred_structure.cpu().numpy())
            
            # Store sample names if available
            if hasattr(batch, 'name'):
                sample_names.extend(batch.name)
            else:
                # For scalar predictions, add single sample name
                if pred_affinity.ndim == 0:
                    sample_names.append(f"sample_{batch_idx}")
                else:
                    sample_names.extend([f"sample_{batch_idx}_{i}" for i in range(len(pred_affinity))])
    
    return predictions, true_values, sample_names, structure_predictions


def save_predictions(predictions, true_values, sample_names, structure_predictions, output_path: str):
    """Save predictions to file."""
    results = {
        'predictions': predictions,
        'true_values': true_values if true_values else None,
        'sample_names': sample_names,
        # 'structure_predictions': structure_predictions if structure_predictions else None
    }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    print(f"Predictions saved to {output_path}")


def main():
    """Main inference function."""
    # Parse arguments
    parser = create_parser_with_config_support()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, help="Output file for predictions (overrides config)")
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config_parser = ConfigParser(args.config)
    else:
        # Use default inference config
        config_parser = ConfigParser("configs/inference_config.yaml")
    
    # Update config with command line arguments
    config_parser.update_from_args(args)
    
    # Create model configuration
    model_config = ModelConfig.from_dict(config_parser.config)
    
    # Get inference parameters
    inference_params = config_parser.get_inference_params()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    # Check if data paths exist
    data_params = config_parser.get_data_params()
    if not os.path.exists(data_params['data_path']):
        raise FileNotFoundError(f"Data path not found: {data_params['data_path']}")
    
    if not os.path.exists(inference_params['meta_path']):
        raise FileNotFoundError(f"Meta path not found: {inference_params['meta_path']}")
    
    # Set device
    device = inference_params.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, model_config)
    print("Model loaded successfully")
    
    # Create dataloader
    print(f"Creating dataloader from: {inference_params['meta_path']}")
    dataloader = create_inference_dataloader_from_config(config_parser)
    print(f"Dataloader created with {len(dataloader)} batches")
    
    # Run inference
    print("Running inference...")
    predictions, true_values, sample_names, structure_predictions = run_inference(model, dataloader, device)
    print(f"Inference complete. Processed {len(predictions)} samples")
    
    # Determine output path
    output_path = args.output or inference_params.get('output_path', 'predictions.json')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save predictions
    save_predictions(predictions, true_values, sample_names, structure_predictions, output_path)
    
    # Print summary statistics
    if predictions:
        pred_array = np.array(predictions)
        print(f"\nPrediction statistics:")
        print(f"  Mean: {pred_array.mean():.4f}")
        print(f"  Std: {pred_array.std():.4f}")
        print(f"  Min: {pred_array.min():.4f}")
        print(f"  Max: {pred_array.max():.4f}")
        
        if true_values:
            true_array = np.array(true_values)
            mae = np.mean(np.abs(pred_array - true_array))
            rmse = np.sqrt(np.mean((pred_array - true_array) ** 2))
            correlation = np.corrcoef(pred_array, true_array)[0, 1]
            print(f"\nEvaluation metrics:")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Correlation: {correlation:.4f}")


if __name__ == "__main__":
    main()