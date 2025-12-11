"""
models/base_mlp.py

Original MLP architecture extracted from training scripts for comparison.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, ELU


def build_base_mlp(input_dim: int) -> tf.keras.Model:
    """
    Builds the original MLP architecture used in the notebooks.
    
    Architecture:
        Input -> Dense(30) -> LeakyReLU -> Dense(60) -> ELU -> Dense(90) -> LeakyReLU -> Dense(1)
    
    Args:
        input_dim: Number of input features
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Dense(30, input_dim=input_dim),
        LeakyReLU(),
        Dense(60),
        ELU(),
        Dense(90),
        LeakyReLU(),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
