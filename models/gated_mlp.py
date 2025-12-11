"""
models/gated_mlp.py

Gated residual MLP architecture inspired by the PINN reference implementation.
Incorporates residual connections and learnable gating mechanism for improved
gradient flow and adaptive feature weighting.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer, Concatenate, LayerNormalization
from tensorflow.keras.initializers import GlorotUniform


class ResidualBlock(Layer):
    """
    A residual block that applies a dense transformation and adds it to the input.
    
    H_out = activation(Dense(H_in)) + H_in
    
    Uses Glorot uniform initialization and optional layer normalization for stability.
    """
    
    def __init__(self, units: int, activation: str = 'tanh', 
                 use_layer_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.dense = None
        self.layer_norm = None
    
    def build(self, input_shape):
        # Use bounded initialization for stability
        self.dense = Dense(
            self.units, 
            activation=self.activation,
            kernel_initializer=GlorotUniform(seed=42),
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )
        if self.use_layer_norm:
            self.layer_norm = LayerNormalization()
        super().build(input_shape)
    
    def call(self, inputs):
        out = self.dense(inputs) + inputs
        if self.use_layer_norm and self.layer_norm is not None:
            out = self.layer_norm(out)
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'use_layer_norm': self.use_layer_norm
        })
        return config


class GatedOptionMLP(Model):
    """
    Gated residual architecture for option pricing.
    
    Based on EuropeanCall_gated3 from the PINN reference implementation.
    
    Architecture:
        - Input projection to hidden dimension
        - N residual blocks with skip connections
        - Dual output paths:
            - Direct path: input -> output (captures linear relationships)
            - Deep path: input -> hidden -> output (captures nonlinearities)
        - Learnable gating weight that combines both paths
    
    Forward pass:
        H = input_projection(x)
        for each residual_layer:
            H = residual_layer(H) + H
        
        y_direct = direct_path(x)
        y_deep = deep_path(H)
        gate_weight = gate([H, x])
        
        output = y_direct + gate_weight * y_deep
    
    Args:
        input_dim: Number of input features
        hidden_width: Width of hidden layers (default: 64)
        n_layers: Number of residual blocks (default: 4)
        activation: Activation function for hidden layers (default: 'tanh')
    """
    
    def __init__(self, input_dim: int, hidden_width: int = 64, n_layers: int = 4,
                 activation: str = 'tanh', use_layer_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.hidden_width = hidden_width
        self.n_layers = n_layers
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        
        # Stable initializer
        init = GlorotUniform(seed=42)
        
        # Input projection: map input features to hidden dimension
        self.input_projection = Dense(
            hidden_width, 
            activation=activation, 
            kernel_initializer=init,
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            name='input_proj'
        )
        
        # Optional layer norm after input projection
        self.input_norm = LayerNormalization() if use_layer_norm else None
        
        # Residual blocks with layer normalization
        self.residual_layers = [
            ResidualBlock(hidden_width, activation=activation, 
                         use_layer_norm=use_layer_norm, name=f'res_block_{i}')
            for i in range(n_layers)
        ]
        
        # Direct path: input -> single output (linear shortcut)
        # Small initialization to prevent dominating early training
        self.direct_path = Dense(
            1, 
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name='direct_path'
        )
        
        # Deep path: hidden state -> single output
        self.deep_path = Dense(
            1, 
            kernel_initializer=init,
            name='deep_path'
        )
        
        # Gating mechanism: learns to weight deep path contribution
        # Initialize bias to make gate start around 0.5
        self.gate_layer = Dense(
            1, 
            activation='sigmoid', 
            kernel_initializer=init,
            bias_initializer=tf.keras.initializers.Constant(0.0),
            name='gate'
        )
        
        # Concatenation layer for gate input
        self.concat = Concatenate(axis=-1)
    
    def call(self, inputs, training=None):
        # Input projection
        H = self.input_projection(inputs)
        if self.input_norm is not None:
            H = self.input_norm(H)
        
        # Pass through residual blocks
        for res_layer in self.residual_layers:
            H = res_layer(H)
        
        # Direct path: input -> output (captures linear relationships)
        y_direct = self.direct_path(inputs)
        
        # Deep path: hidden -> output (captures complex patterns)
        y_deep = self.deep_path(H)
        
        # Compute gating weight based on hidden state and input
        gate_input = self.concat([H, inputs])
        gate_weight = self.gate_layer(gate_input)
        
        # Combine paths: y = y_direct + gate * y_deep
        output = y_direct + gate_weight * y_deep
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'hidden_width': self.hidden_width,
            'n_layers': self.n_layers,
            'activation': self.activation,
            'use_layer_norm': self.use_layer_norm
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_gated_model(input_dim: int, hidden_width: int = 64, n_layers: int = 4,
                      activation: str = 'tanh', learning_rate: float = 0.001,
                      use_layer_norm: bool = True, clipnorm: float = 1.0) -> tf.keras.Model:
    """
    Factory function to build and compile a GatedOptionMLP model.
    
    Args:
        input_dim: Number of input features
        hidden_width: Width of hidden layers (default: 64)
        n_layers: Number of residual blocks (default: 4)
        activation: Activation function (default: 'tanh')
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
        use_layer_norm: Whether to use layer normalization (default: True)
        clipnorm: Gradient clipping norm (default: 1.0)
        
    Returns:
        Compiled GatedOptionMLP model
    """
    model = GatedOptionMLP(
        input_dim=input_dim,
        hidden_width=hidden_width,
        n_layers=n_layers,
        activation=activation,
        use_layer_norm=use_layer_norm
    )
    
    # Use gradient clipping to prevent exploding gradients
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=clipnorm
    )
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Build model by calling it once with dummy input
    dummy_input = tf.zeros((1, input_dim))
    _ = model(dummy_input)
    
    return model
