import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch 

from src.model.model_exp_water import AFFModel_ThreeBody as AFFModel_ExplicitWater
from src.model.model_twobody import AFFModel_TwoBody
from src.model.model_virtual import AFFModel_ThreeBody as AFFModel_VirtualWater
from src.data.datamodule import RLADataModule

from scripts.train.train_utils import CoordinateSaverCallback, DelayedEarlyStopping

def main(args):
    pl.seed_everything(args.seed)

    # === Init wandb (optional disable)

    wandb_logger = WandbLogger(
        mode = args.wandb_mode,
        name=args.run_name,
        project=args.project,
        save_dir=args.output_dir,
        log_model=True
    )
    wandb_logger.experiment.config.update(vars(args))  # log all CLI args

    # === Data module
    if args.small_set: 
        args.train_data_path = 'data/debug.json'
        args.valid_data_path = 'data/debug.json'
        args.test_data_path = 'data/debug.json'
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

    # === Model
    if args.model_type == 'virtual': 
        model = AFFModel_VirtualWater(
            protein_node_dims=(args.protein_scalar_dim, args.protein_vector_dim),
            protein_edge_dims=(args.protein_edge_scalar_dim, args.protein_edge_vector_dim),
            ligand_node_dims=(args.ligand_scalar_dim, 0),
            ligand_edge_dims=(args.ligand_edge_scalar_dim, 0),
            protein_hidden_dims=(args.hidden_scalar_dim, args.hidden_vector_dim),
            virtual_hidden_dims=(args.hidden_scalar_dim, 3),
            ligand_hidden_dims=(args.hidden_scalar_dim, 0),
            num_gvp_layers=args.num_gvp_layers,
            dropout=args.dropout,
            lr=args.learning_rate,
            interaction_mode=args.interaction_mode,  # "hierarchical" or "parallel"
            loss_type=args.loss_type,  # "multitask" or "single"
            predict_str=args.do_structure_prediction,  # whether to predict structure
        )
    elif args.model_type == 'explicit-water':
        model = AFFModel_ExplicitWater(
            protein_node_dims=(args.protein_scalar_dim, args.protein_vector_dim),
            protein_edge_dims=(32, args.protein_edge_vector_dim),
            ligand_node_dims=(args.ligand_scalar_dim, 0),
            ligand_edge_dims=(args.ligand_edge_scalar_dim, 0),
            protein_hidden_dims=(args.hidden_scalar_dim, args.hidden_vector_dim),
            water_hidden_dims=(args.hidden_scalar_dim, 3),
            ligand_hidden_dims=(args.hidden_scalar_dim, 0),
            num_gvp_layers=args.num_gvp_layers,
            dropout=args.dropout,
            lr=args.learning_rate,
            interaction_mode=args.interaction_mode, #"parallel",
            loss_type=args.loss_type 
        )
    elif args.model_type == 'twobody':
        model = AFFModel_TwoBody(
            protein_node_dims=(args.protein_scalar_dim, args.protein_vector_dim),
            protein_edge_dims=(32, args.protein_edge_vector_dim),
            ligand_node_dims=(args.ligand_scalar_dim, 0),
            ligand_edge_dims=(args.ligand_edge_scalar_dim, 0),
            protein_hidden_dims=(args.hidden_scalar_dim, args.hidden_vector_dim),
            ligand_hidden_dims=(args.hidden_scalar_dim, 0),
            num_gvp_layers=args.num_gvp_layers,
            dropout=args.dropout,
            lr=args.learning_rate,
        )
    # === Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='model-{epoch:02d}-{val_loss:.4f}',
        monitor='val_mae',
        mode='min',
        save_top_k=3,
        save_last=True
    )

    early_stop_callback = DelayedEarlyStopping(
    monitor='val_reg_loss',
    mode='min',
    patience=args.patience,
    min_delta=0.01,
    min_epochs_before_stop=80  # delay early stopping until at least 20 epochs
)
    
    lig_coord_saver = CoordinateSaverCallback(
        save_every_n_epochs=args.save_coords_every_n_epochs,  # Save every N epochs
        original_mol2_dir="/home/j2ho/DB/biolip/BioLiP_updated_set/ligand_mol2",
        output_mol2_dir=f"./predicted_mol2/{args.run_name}",
        save_coords_pt=False,  # Also save as .pt files
        separate_epoch_dirs=False)
    
    
    # === Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, lig_coord_saver, early_stop_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        accumulate_grad_batches=args.accum_grad,
        accelerator=args.accelerator,
        strategy=DDPStrategy(find_unused_parameters=True) if args.ddp and args.accelerator == 'gpu' else 'auto',
        devices=args.devices,
        num_nodes=args.num_nodes if hasattr(args, 'num_nodes') else 1,
    )
    debugger = pl.Trainer(
        limit_train_batches=3,
        limit_val_batches=3,
        limit_test_batches=3,
        max_epochs=args.max_epochs,
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
        accelerator='auto',
        devices=args.devices
    )
    if args.debug:
        # Run a single epoch for debugging
        debugger.fit(model, data_module)
        return
    else: 
        trainer.fit(model, data_module)
        trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GVP-based protein-ligand affinity model')

    # === General
    general = parser.add_argument_group('General')
    general.add_argument('--seed', type=int, default=42)
    general.add_argument('--output_dir', type=str, default='./outputs')
    general.add_argument('--disable_wandb', action='store_true')
    general.add_argument('--wandb_mode', type=str, choices=['online', 'offline', 'disabled'], default='online')
    general.add_argument('--project', type=str, default='protein-ligand-affinity')
    general.add_argument('--run_name', type=str, default='gvp_affinity_run')
    general.add_argument('--model_type', type=str, choices=['virtual','twobody','explicit-water'], default='twobody',
                         help='Model type to use: "gvp" for GVP model, "three_body" for three-body model')
    general.add_argument('--debug', action='store_true', help='Enable debugging mode with reduced epochs and batch size')
    # === Data
    data = parser.add_argument_group('Data')
    data.add_argument('--data_path', type=str, default='data/')
    data.add_argument('--train_data_path', type=str, default='data/train.json')
    data.add_argument('--valid_data_path', type=str, default='data/val.json')
    data.add_argument('--test_data_path', type=str, default='data/casp16.json')
    data.add_argument('--small_set', action='store_true', help='Use small dataset for quick testing')
    data.add_argument('--batch_size', type=int, default=64)
    data.add_argument('--num_workers', type=int, default=4)
    data.add_argument('--top_k', type=int, default=30)
    data.add_argument('--crop_size', type=int, default=30)

    # === Training
    train = parser.add_argument_group('Training')
    train.add_argument('--save_coords_every_n_epochs', type=int, default=20,
                       help='Save ligand coordinates every N epochs')
    train.add_argument('--accum_grad', type=int, default=1,
                       help='Number of gradient accumulation steps')

    # === GVP model dims
    model = parser.add_argument_group('Model: Feature Dimensions')
    model.add_argument('--protein_scalar_dim', type=int, default=26)
    model.add_argument('--protein_vector_dim', type=int, default=3)
    model.add_argument('--protein_edge_scalar_dim', type=int, default=41)
    model.add_argument('--protein_edge_vector_dim', type=int, default=1)
    model.add_argument('--ligand_scalar_dim', type=int, default=46)
    model.add_argument('--ligand_edge_scalar_dim', type=int, default=9)

    model.add_argument('--hidden_scalar_dim', type=int, default=196)
    model.add_argument('--hidden_vector_dim', type=int, default=16)
    model.add_argument('--num_gvp_layers', type=int, default=4)
    model.add_argument('--dropout', type=float, default=0.1)
    model.add_argument('--interaction_mode', type=str, choices=['hierarchical', 'parallel'], default='hierarchical')
    model.add_argument('--loss_type', type=str, choices=['multitask', 'single'], default='single') 
    model.add_argument('--do_structure_prediction', action='store_true',
                        help='Whether to predict ligand structure in addition to affinity')
    # === Optimizationtrain_
    optim = parser.add_argument_group('Optimization')
    optim.add_argument('--learning_rate', type=float, default=1e-4)
    optim.add_argument('--max_epochs', type=int, default=150)
    optim.add_argument('--patience', type=int, default=10)
    optim.add_argument('--devices', type=int, default=-1)
    optim.add_argument('--ddp', action='store_true', help='Enable DDP strategy for multi-GPU training')
    args = parser.parse_args()
    devices=torch.cuda.device_count()
    if args.devices == -1:  # use -1 as a sentinel for auto-detect
        if torch.cuda.is_available():
            args.devices = torch.cuda.device_count()
            args.accelerator = 'gpu'
        else:
            args.devices = 1
            args.accelerator = 'cpu'
    else:
        args.accelerator = 'gpu' if torch.cuda.is_available() and args.devices > 0 else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
