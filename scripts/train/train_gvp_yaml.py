"""
Training script using YAML configuration system.
"""

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch

# Import model and data module
from src.model.model_threebody_virtual import AFFModel_ThreeBody as AFFModel_VirtualWater
from src.model.threebody_virtual_glem import AFFModel_ThreeBody as AFFModel_VirtualWater_GLEM
from src.data.datamodule import RLADataModule
from scripts.train.train_utils import CoordinateSaverCallback, DelayedEarlyStopping

# Import config utilities
from scripts.utils.config_parser import ConfigParser, create_parser_with_config_support
from scripts.utils.model_config import ModelConfig


def create_model_from_config(config: ModelConfig):
    """Create model from configuration."""
    model_kwargs = config.to_model_kwargs()
    
    # Check model type and return appropriate model
    if config.model_type == "virtual_glem":
        return AFFModel_VirtualWater_GLEM(**model_kwargs)
    elif config.model_type == "virtual":
        # Default to virtual water model for "virtual" or other types
        return AFFModel_VirtualWater(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}. Supported types are 'virtual' and 'virtual_glem'.")


def create_data_module_from_config(config_parser: ConfigParser) -> RLADataModule:
    """Create data module from configuration."""
    data_params = config_parser.get_data_params()
    return RLADataModule(**data_params)


def create_trainer_from_config(config_parser: ConfigParser, callbacks: list, logger) -> pl.Trainer:
    """Create trainer from configuration."""
    trainer_params = config_parser.get_trainer_params()
    
    trainer = pl.Trainer(
        min_epochs=50,
        max_epochs=trainer_params['max_epochs'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=trainer_params['log_every_n_steps'],
        accumulate_grad_batches=trainer_params['accumulate_grad_batches'],
        accelerator=trainer_params['accelerator'],
        strategy=trainer_params['strategy'],
        devices=trainer_params['devices'],
        num_nodes=trainer_params['num_nodes'],
        gradient_clip_val=trainer_params['gradient_clip_val'],
        gradient_clip_algorithm=trainer_params['gradient_clip_algorithm']
    )
    
    return trainer


def create_callbacks_from_config(config_parser: ConfigParser, run_name: str) -> list:
    """Create callbacks from configuration."""
    logging_params = config_parser.get_logging_params()
    checkpoint_config = logging_params['checkpoint']
    early_stopping_config = logging_params['early_stopping']
    coord_saving_config = logging_params['coordinate_saving']
    
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logging_params['output_dir'], checkpoint_config.get('dirpath', 'checkpoints')),
        filename=checkpoint_config.get('filename', 'model-{epoch:02d}-{val_loss:.4f}'),
        monitor=checkpoint_config.get('monitor', 'val_mae'),
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=checkpoint_config.get('save_last', True)
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if early_stopping_config.get('enabled', False):
        early_stop_callback = DelayedEarlyStopping(
            monitor=early_stopping_config.get('monitor', 'val_reg_loss'),
            mode=early_stopping_config.get('mode', 'min'),
            patience=early_stopping_config.get('patience', 10),
            min_delta=early_stopping_config.get('min_delta', 0.01),
            min_epochs_before_stop=early_stopping_config.get('min_epochs_before_stop', 100)
        )
        callbacks.append(early_stop_callback)
    
    # Coordinate saving callback
    if coord_saving_config.get('enabled', False):
        coord_saver = CoordinateSaverCallback(
            save_every_n_epochs=coord_saving_config.get('save_every_n_epochs', 20),
            original_mol2_dir=coord_saving_config.get('original_mol2_dir', ''),
            original_pdb_dir=coord_saving_config.get('original_pdb_dir', ''),
            output_dir=os.path.join(coord_saving_config.get('output_dir', './predicted_str'), run_name),
            save_coords_pt=coord_saving_config.get('save_coords_pt', False),
            separate_epoch_dirs=coord_saving_config.get('separate_epoch_dirs', False)
        )
        callbacks.append(coord_saver)
    
    return callbacks


def create_logger_from_config(config_parser: ConfigParser) -> WandbLogger:
    """Create logger from configuration."""
    logging_params = config_parser.get_logging_params()
    
    wandb_logger = WandbLogger(
        mode=logging_params['wandb_mode'],
        name=logging_params['run_name'],
        project=logging_params['project'],
        save_dir=logging_params['output_dir'],
        log_model=True
    )
    
    return wandb_logger


def main():
    """Main training function."""
    # Parse arguments
    parser = create_parser_with_config_support()
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config_parser = ConfigParser(args.config)
    else:
        # Use default config
        config_parser = ConfigParser("configs/model_config.yaml")
    
    # Update config with command line arguments
    config_parser.update_from_args(args)
    
    # Create model configuration
    model_config = ModelConfig.from_dict(config_parser.config)
    
    # Set seed
    training_config = config_parser.config.get('training', {})
    seed = training_config.get('seed', 42)
    pl.seed_everything(seed)
    torch.autograd.set_detect_anomaly(True)
    
    # Create logger
    wandb_logger = create_logger_from_config(config_parser)
    
    # Log all configuration
    wandb_logger.experiment.config.update(config_parser.config)
    if args.config:
        wandb_logger.experiment.config.update({"config_file": args.config})
    
    # Handle debug mode
    if args.debug:
        # Override for debugging
        config_parser.config['data']['train_data_path'] = 'data/debug.json'
        config_parser.config['data']['valid_data_path'] = 'data/debug.json'
        config_parser.config['data']['test_data_path'] = 'data/debug.json'
        config_parser.config['data']['batch_size'] = 8
    
    # Create data module
    data_module = create_data_module_from_config(config_parser)
    
    # Create model
    model = create_model_from_config(model_config)
    
    # Create callbacks
    logging_params = config_parser.get_logging_params()
    run_name = logging_params['run_name']
    callbacks = create_callbacks_from_config(config_parser, run_name)
    
    # Create trainer
    trainer = create_trainer_from_config(config_parser, callbacks, wandb_logger)
    
    # Create debug trainer if needed
    if args.debug:
        debugger = pl.Trainer(
            limit_train_batches=3,
            limit_val_batches=3,
            limit_test_batches=3,
            max_epochs=5,
            logger=False,
            enable_checkpointing=False,
            log_every_n_steps=1,
            accelerator='auto',
            devices=1
        )
        debugger.fit(model, data_module)
        return
    
    # Ensure output directory exists
    os.makedirs(logging_params['output_dir'], exist_ok=True)
    
    # Save the configuration used for this run
    config_save_path = os.path.join(logging_params['output_dir'], f"{run_name}_config.yaml")
    config_parser.save_config(config_save_path)
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Optionally run test
    # trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()