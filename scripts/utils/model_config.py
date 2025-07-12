"""
Model configuration classes for better parameter organization.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple


@dataclass
class NodeDimensions:
    """Configuration for node dimensions."""
    scalar: int
    vector: int
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convert to tuple format expected by model."""
        return (self.scalar, self.vector)


@dataclass
class EdgeDimensions:
    """Configuration for edge dimensions."""
    scalar: int
    vector: int
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convert to tuple format expected by model."""
        return (self.scalar, self.vector)


@dataclass
class HiddenDimensions:
    """Configuration for hidden layer dimensions."""
    protein_scalar: int = 196
    protein_vector: int = 16
    virtual_scalar: int = 196
    virtual_vector: int = 3
    ligand_scalar: int = 196
    ligand_vector: int = 3
    
    @property
    def protein_dims(self) -> Tuple[int, int]:
        """Get protein hidden dimensions as tuple."""
        return (self.protein_scalar, self.protein_vector)
    
    @property
    def virtual_dims(self) -> Tuple[int, int]:
        """Get virtual hidden dimensions as tuple."""
        return (self.virtual_scalar, self.virtual_vector)
    
    @property
    def ligand_dims(self) -> Tuple[int, int]:
        """Get ligand hidden dimensions as tuple."""
        return (self.ligand_scalar, self.ligand_vector)


@dataclass
class StructurePredictionConfig:
    """Configuration for structure prediction."""
    enabled: bool = False
    model_type: str = "egnn"  # "mlp" or "egnn"
    loss_weight: float = 0.3
    loss_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.loss_params is None:
            self.loss_params = {
                "loss_weight": self.loss_weight,
                "loss_type": "mse",
                "ligand_weight": 1.0,
                "sidechain_weight": 1.0,
                "use_distance_loss": True,
                "ligand_distance_weight": 0.3,
                "sidechain_distance_weight": 0.1,
                "distance_loss_type": "mse",
                "distance_cutoff": 5.0
            }


@dataclass
class AffinityLossConfig:
    """Configuration for affinity loss."""
    beta: float = 1.0
    extreme_weight: float = 1.5
    ranking_weight: float = 0.5


@dataclass
class MultitaskLossConfig:
    """Configuration for multitask loss."""
    thresholds: List[float] = field(default_factory=lambda: [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
    regression_weight: float = 0.75
    category_penalty_weight: float = 0.15
    extreme_penalty_weight: float = 0.0
    pearson_penalty_weight: float = 0.1
    relative_error_weight: float = 0.1
    extreme_boost_low: float = 1.3
    extreme_boost_high: float = 1.4


@dataclass
class LossSchedulingConfig:
    """Configuration for loss scheduling."""
    enabled: bool = True
    structure_warmup_epochs: int = 15
    affinity_warmup_epochs: int = 25
    structure_max_weight: float = 1.0
    affinity_max_weight: float = 1.0
    affinity_min_weight: float = 1.0
    
    @property
    def scheduling_params(self) -> Dict[str, Any]:
        """Get loss scheduling parameters for model initialization."""
        return {
            'enabled': self.enabled,
            'structure_warmup_epochs': self.structure_warmup_epochs,
            'affinity_warmup_epochs': self.affinity_warmup_epochs,
            'structure_max_weight': self.structure_max_weight,
            'affinity_max_weight': self.affinity_max_weight,
            'affinity_min_weight': self.affinity_min_weight,
        }


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    type: str = "single"  # "single" or "multitask"
    affinity_loss: AffinityLossConfig = field(default_factory=AffinityLossConfig)
    multitask_loss: MultitaskLossConfig = field(default_factory=MultitaskLossConfig)
    scheduling: LossSchedulingConfig = field(default_factory=LossSchedulingConfig)
    
    @property
    def loss_params(self) -> Dict[str, Any]:
        """Get loss parameters based on loss type."""
        if self.type == "single":
            return {
                "beta": self.affinity_loss.beta,
                "extreme_weight": self.affinity_loss.extreme_weight,
                "ranking_weight": self.affinity_loss.ranking_weight
            }
        else:  # multitask
            return {
                "thresholds": self.multitask_loss.thresholds,
                "regression_weight": self.multitask_loss.regression_weight,
                "category_penalty_weight": self.multitask_loss.category_penalty_weight,
                "extreme_penalty_weight": self.multitask_loss.extreme_penalty_weight,
                "pearson_penalty_weight": self.multitask_loss.pearson_penalty_weight,
                "relative_error_weight": self.multitask_loss.relative_error_weight,
                "extreme_boost_low": self.multitask_loss.extreme_boost_low,
                "extreme_boost_high": self.multitask_loss.extreme_boost_high
            }


@dataclass
class ModelConfig:
    """Main model configuration."""
    # Model type
    model_type: str = "virtual"
    
    # Node dimensions
    protein_node_dims: NodeDimensions = field(default_factory=lambda: NodeDimensions(26, 3))
    virtual_node_dims: NodeDimensions = field(default_factory=lambda: NodeDimensions(26, 3))
    ligand_node_dims: NodeDimensions = field(default_factory=lambda: NodeDimensions(46, 0))
    
    # Edge dimensions
    protein_edge_dims: EdgeDimensions = field(default_factory=lambda: EdgeDimensions(41, 1))
    ligand_edge_dims: EdgeDimensions = field(default_factory=lambda: EdgeDimensions(9, 0))
    
    # Hidden dimensions
    hidden_dims: HiddenDimensions = field(default_factory=HiddenDimensions)
    
    # Model architecture
    num_gvp_layers: int = 3
    interaction_mode: str = "hierarchical"  # "hierarchical" or "parallel"
    
    # Structure prediction
    structure_prediction: StructurePredictionConfig = field(default_factory=StructurePredictionConfig)
    
    # Loss configuration
    loss: LossConfig = field(default_factory=LossConfig)
    
    # Training parameters
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    
    def to_model_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs suitable for model initialization."""
        return {
            # Node dimensions
            'protein_node_dims': self.protein_node_dims.to_tuple(),
            'virtual_node_dims': self.virtual_node_dims.to_tuple(),
            'ligand_node_dims': self.ligand_node_dims.to_tuple(),
            
            # Edge dimensions
            'protein_edge_dims': self.protein_edge_dims.to_tuple(),
            'ligand_edge_dims': self.ligand_edge_dims.to_tuple(),
            
            # Hidden dimensions
            'protein_hidden_dims': self.hidden_dims.protein_dims,
            'virtual_hidden_dims': self.hidden_dims.virtual_dims,
            'ligand_hidden_dims': self.hidden_dims.ligand_dims,
            
            # Model architecture
            'num_gvp_layers': self.num_gvp_layers,
            'dropout': float(self.dropout),
            'lr': float(self.learning_rate),
            'weight_decay': float(self.weight_decay),
            'interaction_mode': self.interaction_mode,
            
            # Structure prediction
            'predict_str': self.structure_prediction.enabled,
            'str_model_type': self.structure_prediction.model_type,
            'str_loss_params': self.structure_prediction.loss_params,
            
            # Loss configuration
            'loss_type': self.loss.type,
            'loss_params': self.loss.loss_params,
            
            # Loss scheduling (grouped)
            'loss_scheduling_params': self.loss.scheduling.scheduling_params,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary (e.g., from YAML)."""
        # Extract model config
        model_config = config_dict.get('model', {})
        loss_config = config_dict.get('loss', {})
        training_config = config_dict.get('training', {})
        
        # Create node dimensions
        protein_node_dims = NodeDimensions(
            scalar=model_config.get('protein_node_dims', {}).get('scalar', 26),
            vector=model_config.get('protein_node_dims', {}).get('vector', 3)
        )
        
        virtual_node_dims = NodeDimensions(
            scalar=model_config.get('virtual_node_dims', {}).get('scalar', 26),
            vector=model_config.get('virtual_node_dims', {}).get('vector', 3)
        )
        
        ligand_node_dims = NodeDimensions(
            scalar=model_config.get('ligand_node_dims', {}).get('scalar', 46),
            vector=model_config.get('ligand_node_dims', {}).get('vector', 0)
        )
        
        # Create edge dimensions
        protein_edge_dims = EdgeDimensions(
            scalar=model_config.get('protein_edge_dims', {}).get('scalar', 41),
            vector=model_config.get('protein_edge_dims', {}).get('vector', 1)
        )
        
        ligand_edge_dims = EdgeDimensions(
            scalar=model_config.get('ligand_edge_dims', {}).get('scalar', 9),
            vector=model_config.get('ligand_edge_dims', {}).get('vector', 0)
        )
        
        # Create hidden dimensions
        hidden_dims_dict = model_config.get('hidden_dims', {})
        hidden_dims = HiddenDimensions(
            protein_scalar=hidden_dims_dict.get('protein_scalar', 196),
            protein_vector=hidden_dims_dict.get('protein_vector', 16),
            virtual_scalar=hidden_dims_dict.get('virtual_scalar', 196),
            virtual_vector=hidden_dims_dict.get('virtual_vector', 3),
            ligand_scalar=hidden_dims_dict.get('ligand_scalar', 196),
            ligand_vector=hidden_dims_dict.get('ligand_vector', 3)
        )
        
        # Create structure prediction config
        str_config = model_config.get('structure_prediction', {})
        structure_prediction = StructurePredictionConfig(
            enabled=str_config.get('enabled', False),
            model_type=str_config.get('model_type', 'egnn'),
            loss_weight=str_config.get('loss_weight', 0.3),
            loss_params=str_config.get('loss_params', None)
        )
        
        # Create loss config
        affinity_loss = AffinityLossConfig(
            beta=loss_config.get('affinity_loss', {}).get('beta', 1.0),
            extreme_weight=loss_config.get('affinity_loss', {}).get('extreme_weight', 1.5),
            ranking_weight=loss_config.get('affinity_loss', {}).get('ranking_weight', 0.5)
        )
        
        multitask_loss_dict = loss_config.get('multitask_loss', {})
        multitask_loss = MultitaskLossConfig(
            thresholds=multitask_loss_dict.get('thresholds', [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]),
            regression_weight=multitask_loss_dict.get('regression_weight', 0.75),
            category_penalty_weight=multitask_loss_dict.get('category_penalty_weight', 0.15),
            extreme_penalty_weight=multitask_loss_dict.get('extreme_penalty_weight', 0.0),
            pearson_penalty_weight=multitask_loss_dict.get('pearson_penalty_weight', 0.1),
            relative_error_weight=multitask_loss_dict.get('relative_error_weight', 0.1),
            extreme_boost_low=multitask_loss_dict.get('extreme_boost_low', 1.3),
            extreme_boost_high=multitask_loss_dict.get('extreme_boost_high', 1.4)
        )
        
        scheduling_dict = loss_config.get('scheduling', {})
        loss_scheduling = LossSchedulingConfig(
            enabled=scheduling_dict.get('enabled', True),
            structure_warmup_epochs=scheduling_dict.get('structure_warmup_epochs', 15),
            affinity_warmup_epochs=scheduling_dict.get('affinity_warmup_epochs', 25),
            structure_max_weight=scheduling_dict.get('structure_max_weight', 1.0),
            affinity_max_weight=scheduling_dict.get('affinity_max_weight', 1.0),
            affinity_min_weight=scheduling_dict.get('affinity_min_weight', 1.0)
        )
        
        loss = LossConfig(
            type=loss_config.get('type', 'single'),
            affinity_loss=affinity_loss,
            multitask_loss=multitask_loss,
            scheduling=loss_scheduling
        )
        
        return cls(
            model_type=model_config.get('model_type', 'virtual'),
            protein_node_dims=protein_node_dims,
            virtual_node_dims=virtual_node_dims,
            ligand_node_dims=ligand_node_dims,
            protein_edge_dims=protein_edge_dims,
            ligand_edge_dims=ligand_edge_dims,
            hidden_dims=hidden_dims,
            num_gvp_layers=model_config.get('num_gvp_layers', 3),
            interaction_mode=model_config.get('interaction_mode', 'hierarchical'),
            structure_prediction=structure_prediction,
            loss=loss,
            dropout=training_config.get('dropout', 0.1),
            learning_rate=training_config.get('learning_rate', 1e-3),
            weight_decay=training_config.get('optimizer', {}).get('weight_decay', 0.01)
        )