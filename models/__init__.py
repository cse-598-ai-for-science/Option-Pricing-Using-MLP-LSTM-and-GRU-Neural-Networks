"""
models/__init__.py

Neural network architectures for option pricing.
"""

from .gated_mlp import GatedOptionMLP, build_gated_model
from .base_mlp import build_base_mlp
from .physics_loss import (
    PINNLoss,
    compute_bs_residual,
    compute_bs_residual_from_features,
    sample_physics_points,
    sample_physics_features,
    sample_boundary_points,
    european_call_payoff,
    european_put_payoff,
)

__all__ = [
    'GatedOptionMLP', 
    'build_gated_model', 
    'build_base_mlp',
    'PINNLoss',
    'compute_bs_residual',
    'compute_bs_residual_from_features',
    'sample_physics_points',
    'sample_physics_features',
    'sample_boundary_points',
    'european_call_payoff',
    'european_put_payoff',
]
