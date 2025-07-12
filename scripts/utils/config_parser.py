"""
Configuration parser for loading YAML config files and converting them to model parameters.
"""

import yaml
import argparse
from typing import Dict, Any, Optional
from pathlib import Path
import os


class ConfigParser:
    """Parse YAML configuration files and convert to model parameters."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config parser.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        return self.config
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Extract model parameters from the configuration.
        
        Returns:
            Dictionary of model parameters suitable for AFFModel_ThreeBody.__init__()
        """
        if not self.config:
            raise ValueError("No configuration loaded. Call load_config() first.")
        
        model_config = self.config.get('model', {})
        loss_config = self.config.get('loss', {})
        
        # Extract node dimensions
        protein_node_dims = (
            model_config.get('protein_node_dims', {}).get('scalar', 26),
            model_config.get('protein_node_dims', {}).get('vector', 3)
        )
        
        virtual_node_dims = (
            model_config.get('virtual_node_dims', {}).get('scalar', 26),
            model_config.get('virtual_node_dims', {}).get('vector', 3)
        )
        
        ligand_node_dims = (
            model_config.get('ligand_node_dims', {}).get('scalar', 46),
            model_config.get('ligand_node_dims', {}).get('vector', 0)
        )
        
        # Extract edge dimensions
        protein_edge_dims = (
            model_config.get('protein_edge_dims', {}).get('scalar', 41),
            model_config.get('protein_edge_dims', {}).get('vector', 1)
        )
        
        ligand_edge_dims = (
            model_config.get('ligand_edge_dims', {}).get('scalar', 9),
            model_config.get('ligand_edge_dims', {}).get('vector', 0)
        )
        
        # Extract hidden dimensions
        hidden_dims = model_config.get('hidden_dims', {})
        protein_hidden_dims = (
            hidden_dims.get('protein_scalar', 196),
            hidden_dims.get('protein_vector', 16)
        )
        
        virtual_hidden_dims = (
            hidden_dims.get('virtual_scalar', 196),
            hidden_dims.get('virtual_vector', 3)
        )
        
        ligand_hidden_dims = (
            hidden_dims.get('ligand_scalar', 196),
            hidden_dims.get('ligand_vector', 3)
        )
        
        # Extract structure prediction parameters
        str_config = model_config.get('structure_prediction', {})
        
        # Extract loss parameters
        affinity_loss_params = loss_config.get('affinity_loss', {})
        multitask_loss_params = loss_config.get('multitask_loss', {})
        scheduling_params = loss_config.get('scheduling', {})
        
        # Extract training parameters
        training_config = self.config.get('training', {})
        
        # Build model parameters
        model_params = {
            # Node dimensions
            'protein_node_dims': protein_node_dims,
            'virtual_node_dims': virtual_node_dims,
            'ligand_node_dims': ligand_node_dims,
            
            # Edge dimensions
            'protein_edge_dims': protein_edge_dims,
            'ligand_edge_dims': ligand_edge_dims,
            
            # Hidden dimensions
            'protein_hidden_dims': protein_hidden_dims,
            'virtual_hidden_dims': virtual_hidden_dims,
            'ligand_hidden_dims': ligand_hidden_dims,
            
            # Model architecture
            'num_gvp_layers': model_config.get('num_gvp_layers', 3),
            'dropout': model_config.get('dropout', 0.1),
            'interaction_mode': model_config.get('interaction_mode', 'hierarchical'),
            
            # Training parameters
            'lr': training_config.get('learning_rate', 1e-3),
            
            # Structure prediction
            'predict_str': str_config.get('enabled', False),
            'str_model_type': str_config.get('model_type', 'egnn'),
            'str_loss_weight': str_config.get('loss_weight', 0.3),
            'str_loss_params': str_config.get('loss_params', None),
            
            # Loss configuration
            'loss_type': loss_config.get('type', 'single'),
            'loss_params': affinity_loss_params if loss_config.get('type', 'single') == 'single' else multitask_loss_params,
            
            # Loss scheduling
            'use_loss_scheduling': scheduling_params.get('enabled', True),
            'str_warmup_epochs': scheduling_params.get('structure_warmup_epochs', 15),
            'aff_warmup_epochs': scheduling_params.get('affinity_warmup_epochs', 25),
            'str_max_weight': scheduling_params.get('structure_max_weight', 1.0),
            'aff_max_weight': scheduling_params.get('affinity_max_weight', 1.0),
        }
        
        return model_params
    
    def get_trainer_params(self) -> Dict[str, Any]:
        """
        Extract trainer parameters from the configuration.
        
        Returns:
            Dictionary of trainer parameters
        """
        if not self.config:
            raise ValueError("No configuration loaded. Call load_config() first.")
        
        training_config = self.config.get('training', {})
        hardware_config = self.config.get('hardware', {})
        logging_config = self.config.get('logging', {})
        
        trainer_params = {
            'max_epochs': training_config.get('max_epochs', 1000),
            'gradient_clip_val': training_config.get('gradient_clip_val', 1.0),
            'gradient_clip_algorithm': training_config.get('gradient_clip_algorithm', 'norm'),
            'accumulate_grad_batches': hardware_config.get('accumulate_grad_batches', 1),
            'accelerator': hardware_config.get('accelerator', 'auto'),
            'devices': hardware_config.get('devices', -1),
            'strategy': hardware_config.get('strategy', 'ddp_find_unused_parameters_true'),
            'num_nodes': hardware_config.get('num_nodes', 1),
            'log_every_n_steps': logging_config.get('log_every_n_steps', 1),
        }
        
        return trainer_params
    
    def get_data_params(self) -> Dict[str, Any]:
        """
        Extract data module parameters from the configuration.
        
        Returns:
            Dictionary of data module parameters
        """
        if not self.config:
            raise ValueError("No configuration loaded. Call load_config() first.")
        
        data_config = self.config.get('data', {})
        
        data_params = {
            'data_path': data_config.get('data_path', 'data/'),
            'train_meta_path': data_config.get('train_data_path', 'data/train.json'),
            'val_meta_path': data_config.get('valid_data_path', 'data/val.json'),
            'test_meta_path': data_config.get('test_data_path', 'data/casp16.json'),
            'glem_data_path': data_config.get('glem_data_path', None),
            'batch_size': data_config.get('batch_size', 64),
            'num_workers': data_config.get('num_workers', 4),
            'top_k': data_config.get('top_k', 30),
            'crop_size': data_config.get('crop_size', 30),
        }
        
        return data_params
    
    def get_logging_params(self) -> Dict[str, Any]:
        """
        Extract logging parameters from the configuration.
        
        Returns:
            Dictionary of logging parameters
        """
        if not self.config:
            raise ValueError("No configuration loaded. Call load_config() first.")
        
        logging_config = self.config.get('logging', {})
        
        logging_params = {
            'project': logging_config.get('project', 'protein-ligand-affinity'),
            'run_name': logging_config.get('run_name', 'gvp_affinity_run'),
            'wandb_mode': logging_config.get('wandb_mode', 'online'),
            'output_dir': logging_config.get('output_dir', './outputs'),
            'checkpoint': logging_config.get('checkpoint', {}),
            'early_stopping': logging_config.get('early_stopping', {}),
            'coordinate_saving': logging_config.get('coordinate_saving', {}),
        }
        
        return logging_params
    
    def get_inference_params(self) -> Dict[str, Any]:
        """
        Extract inference parameters from the configuration.
        
        Returns:
            Dictionary of inference parameters
        """
        if not self.config:
            raise ValueError("No configuration loaded. Call load_config() first.")
        
        inference_config = self.config.get('inference', {})
        
        inference_params = {
            'meta_path': inference_config.get('meta_path', 'data/test.json'),
            'output_path': inference_config.get('output_path', 'results/predictions.json'),
            'device': inference_config.get('device', 'cuda'),
            'batch_size': inference_config.get('batch_size', 1),
            'num_workers': inference_config.get('num_workers', 4),
            'save_structures': inference_config.get('save_structures', False),
            'structure_output_dir': inference_config.get('structure_output_dir', 'results/structures/'),
        }
        
        return inference_params
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        """
        Update configuration with command line arguments.
        
        Args:
            args: Command line arguments from argparse
        """
        if not self.config:
            self.config = {}
        
        # Update with any provided arguments
        if hasattr(args, 'learning_rate') and args.learning_rate is not None:
            if 'training' not in self.config:
                self.config['training'] = {}
            self.config['training']['learning_rate'] = args.learning_rate
        
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            if 'data' not in self.config:
                self.config['data'] = {}
            self.config['data']['batch_size'] = args.batch_size
        
        if hasattr(args, 'max_epochs') and args.max_epochs is not None:
            if 'training' not in self.config:
                self.config['training'] = {}
            self.config['training']['max_epochs'] = args.max_epochs
        
        if hasattr(args, 'run_name') and args.run_name is not None:
            if 'logging' not in self.config:
                self.config['logging'] = {}
            self.config['logging']['run_name'] = args.run_name
        
        # Add more argument updates as needed
    
    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to a YAML file.
        
        Args:
            output_path: Path to save the configuration
        """
        if not self.config:
            raise ValueError("No configuration to save. Load a config first.")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)


def create_parser_with_config_support() -> argparse.ArgumentParser:
    """
    Create an argument parser that supports both command line arguments and YAML config files.
    
    Returns:
        ArgumentParser with config file support
    """
    parser = argparse.ArgumentParser(description='Train GVP-based protein-ligand affinity model')
    
    # Config file argument
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    
    # Keep some essential CLI arguments for override capability
    parser.add_argument('--learning_rate', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--max_epochs', type=int, help='Maximum epochs (overrides config)')
    parser.add_argument('--run_name', type=str, help='Run name (overrides config)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--output_dir', type=str, help='Output directory (overrides config)')
    
    return parser


if __name__ == "__main__":
    # Example usage
    config_parser = ConfigParser("configs/model_config.yaml")
    
    print("Model parameters:")
    model_params = config_parser.get_model_params()
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    
    print("\nTrainer parameters:")
    trainer_params = config_parser.get_trainer_params()
    for key, value in trainer_params.items():
        print(f"  {key}: {value}")