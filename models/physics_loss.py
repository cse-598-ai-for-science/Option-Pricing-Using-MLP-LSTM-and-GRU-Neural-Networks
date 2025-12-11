"""
models/physics_loss.py

Physics-Informed Neural Network loss components for option pricing.
Implements Black-Scholes PDE residual computation using TensorFlow's GradientTape.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional


def compute_bs_residual(model: tf.keras.Model, t: tf.Tensor, S: tf.Tensor,
                        r: float, sigma: float) -> tf.Tensor:
    """
    Compute the Black-Scholes PDE residual for a batch of (t, S) points.
    
    The Black-Scholes PDE for a derivative V(t, S):
        dV/dt + 0.5 * sigma^2 * S^2 * d2V/dS2 + r * S * dV/dS - r * V = 0
    
    Args:
        model: Neural network that takes [t, S] as input and outputs V
        t: Time values, shape (batch_size, 1)
        S: Spot price values, shape (batch_size, 1)
        r: Risk-free interest rate (scalar)
        sigma: Volatility (scalar)
        
    Returns:
        PDE residual tensor, shape (batch_size, 1). Should be close to zero
        if the model satisfies the Black-Scholes equation.
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([t, S])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([t, S])
            # Combine inputs for model
            x = tf.concat([t, S], axis=1)
            V = model(x, training=True)
        
        # First derivatives
        dV_dt = tape1.gradient(V, t)
        dV_dS = tape1.gradient(V, S)
    
    # Second derivative
    d2V_dS2 = tape2.gradient(dV_dS, S)
    
    # Clean up persistent tapes
    del tape1
    del tape2
    
    # Handle None gradients (can happen if model output is independent of input)
    if dV_dt is None:
        dV_dt = tf.zeros_like(V)
    if dV_dS is None:
        dV_dS = tf.zeros_like(V)
    if d2V_dS2 is None:
        d2V_dS2 = tf.zeros_like(V)
    
    # Black-Scholes PDE residual
    # dV/dt + 0.5 * sigma^2 * S^2 * d2V/dS2 + r * S * dV/dS - r * V = 0
    residual = (dV_dt + 
                0.5 * (sigma ** 2) * (S ** 2) * d2V_dS2 + 
                r * S * dV_dS - 
                r * V)
    
    return residual


def compute_bs_residual_from_features(model: tf.keras.Model, 
                                       maturity: tf.Tensor,
                                       k_over_s: tf.Tensor,
                                       r_tensor: tf.Tensor,
                                       iv: tf.Tensor,
                                       cond_vol: tf.Tensor,
                                       r: float,
                                       sigma: float,
                                       extra_features: tf.Tensor = None) -> tf.Tensor:
    """
    Compute Black-Scholes PDE residual using the project's feature representation.
    
    This version works with the normalized feature space used in training:
    [r, K_over_S, Maturity, IV, cond_vol, ...extra_features]
    
    The PDE is adapted for the normalized representation where we track
    derivatives with respect to Maturity (proxy for time) and K/S ratio.
    
    Note: This is an approximation since the exact transformation from
    (t, S) space to feature space involves non-trivial Jacobians.
    
    Args:
        model: Neural network with 5+ input features
        maturity: Time to maturity, shape (batch_size, 1)
        k_over_s: Strike/Spot ratio, shape (batch_size, 1)
        r_tensor: Risk-free rate tensor, shape (batch_size, 1)
        iv: Implied volatility, shape (batch_size, 1)
        cond_vol: Conditional volatility, shape (batch_size, 1)
        r: Risk-free rate scalar for PDE
        sigma: Volatility scalar for PDE
        extra_features: Optional tensor of extra features, shape (batch_size, n_extra)
        
    Returns:
        Approximate PDE residual tensor
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([maturity, k_over_s])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([maturity, k_over_s])
            # Combine features (base 5 features)
            base_features = [r_tensor, k_over_s, maturity, iv, cond_vol]
            if extra_features is not None:
                x = tf.concat(base_features + [extra_features], axis=1)
            else:
                x = tf.concat(base_features, axis=1)
            V = model(x, training=True)
        
        # Derivatives w.r.t. maturity (proxy for time derivative)
        # Note: Maturity = T - t, so dV/dMaturity = -dV/dt
        dV_dT = tape1.gradient(V, maturity)
        # Derivative w.r.t. K/S ratio (related to S derivative)
        dV_dKS = tape1.gradient(V, k_over_s)
    
    # Second derivative w.r.t. K/S
    d2V_dKS2 = tape2.gradient(dV_dKS, k_over_s)
    
    del tape1
    del tape2
    
    # Handle None gradients
    if dV_dT is None:
        dV_dT = tf.zeros_like(V)
    if dV_dKS is None:
        dV_dKS = tf.zeros_like(V)
    if d2V_dKS2 is None:
        d2V_dKS2 = tf.zeros_like(V)
    
    # Approximate PDE residual in feature space
    # This is a simplified version - the exact transformation would require
    # the Jacobian of the coordinate change from (t, S) to (Maturity, K/S)
    # dV/dt = -dV/dMaturity (since Maturity = T - t)
    # For K/S derivative: if K is fixed, d(K/S)/dS = -K/S^2, so dV/dS involves dV/d(K/S)
    
    # Simplified residual (approximate physics constraint)
    residual = (-dV_dT +  # Note: negative because Maturity = T - t
                0.5 * (sigma ** 2) * d2V_dKS2 + 
                r * dV_dKS - 
                r * V)
    
    return residual


def sample_physics_points(n_samples: int, t_range: Tuple[float, float],
                          S_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample random (t, S) points for physics loss evaluation.
    
    Args:
        n_samples: Number of points to sample
        t_range: (t_min, t_max) time range
        S_range: (S_min, S_max) spot price range
        
    Returns:
        Tuple of (t, S) arrays, each shape (n_samples, 1)
    """
    t = np.random.uniform(t_range[0], t_range[1], (n_samples, 1)).astype(np.float32)
    S = np.random.uniform(S_range[0], S_range[1], (n_samples, 1)).astype(np.float32)
    return t, S


def sample_physics_features(n_samples: int, 
                            maturity_range: Tuple[float, float] = (0.01, 2.0),
                            k_over_s_range: Tuple[float, float] = (0.5, 1.5),
                            r_range: Tuple[float, float] = (0.0, 0.1),
                            iv_range: Tuple[float, float] = (0.1, 0.8),
                            cond_vol_range: Tuple[float, float] = (0.1, 0.5),
                            extra_features: list = None) -> dict:
    """
    Sample random feature points for physics loss in feature space.
    
    Args:
        n_samples: Number of points to sample
        maturity_range: (min, max) for time to maturity
        k_over_s_range: (min, max) for strike/spot ratio
        r_range: (min, max) for risk-free rate
        iv_range: (min, max) for implied volatility
        cond_vol_range: (min, max) for conditional volatility
        extra_features: List of extra feature names to include (e.g., ['EPU', 'pos', 'neu', 'neg'])
        
    Returns:
        Dictionary with feature arrays, each shape (n_samples, 1)
    """
    # Default ranges for extra features
    extra_ranges = {
        'EPU': (50.0, 300.0),      # EPU index typical range
        'pos': (0.0, 1.0),          # Sentiment scores are normalized
        'neu': (0.0, 1.0),
        'neg': (0.0, 1.0),
    }
    
    result = {
        'maturity': np.random.uniform(*maturity_range, (n_samples, 1)).astype(np.float32),
        'k_over_s': np.random.uniform(*k_over_s_range, (n_samples, 1)).astype(np.float32),
        'r': np.random.uniform(*r_range, (n_samples, 1)).astype(np.float32),
        'iv': np.random.uniform(*iv_range, (n_samples, 1)).astype(np.float32),
        'cond_vol': np.random.uniform(*cond_vol_range, (n_samples, 1)).astype(np.float32),
    }
    
    # Add extra features if specified
    if extra_features:
        for feat in extra_features:
            feat_range = extra_ranges.get(feat, (0.0, 1.0))  # Default to [0, 1]
            result[feat] = np.random.uniform(*feat_range, (n_samples, 1)).astype(np.float32)
    
    return result


def european_call_payoff(S: tf.Tensor, K: float) -> tf.Tensor:
    """Terminal condition for European call: max(S - K, 0)"""
    return tf.maximum(S - K, 0.0)


def european_put_payoff(S: tf.Tensor, K: float) -> tf.Tensor:
    """Terminal condition for European put: max(K - S, 0)"""
    return tf.maximum(K - S, 0.0)


def sample_boundary_points(n_samples: int, K: float, T: float,
                           S_range: Tuple[float, float]) -> dict:
    """
    Sample boundary condition points for PINN training.
    
    Returns points for:
    - IVP (Initial Value Problem): t = T (at expiry)
    - BVP1 (Boundary Value Problem 1): S = S_min
    - BVP2 (Boundary Value Problem 2): S = S_max
    
    Args:
        n_samples: Number of samples per boundary
        K: Strike price
        T: Time to maturity
        S_range: (S_min, S_max)
        
    Returns:
        Dictionary containing boundary points and expected values
    """
    S_min, S_max = S_range
    
    # IVP: at expiry (t = T), option value = payoff
    ivp_t = np.full((n_samples, 1), T, dtype=np.float32)
    ivp_S = np.random.uniform(S_min, S_max, (n_samples, 1)).astype(np.float32)
    ivp_V = np.maximum(ivp_S - K, 0).astype(np.float32)  # Call payoff
    
    # BVP1: S = 0, call value = 0
    bvp1_t = np.random.uniform(0, T, (n_samples, 1)).astype(np.float32)
    bvp1_S = np.full((n_samples, 1), S_min, dtype=np.float32)
    bvp1_V = np.zeros((n_samples, 1), dtype=np.float32)
    
    # BVP2: S = S_max (deep ITM), call value approx S - K*exp(-r*(T-t))
    # Simplified: just use S - K as approximation for deep ITM
    bvp2_t = np.random.uniform(0, T, (n_samples, 1)).astype(np.float32)
    bvp2_S = np.full((n_samples, 1), S_max, dtype=np.float32)
    bvp2_V = (bvp2_S - K).astype(np.float32)  # Approximation for deep ITM
    
    return {
        'ivp': {'t': ivp_t, 'S': ivp_S, 'V': ivp_V},
        'bvp1': {'t': bvp1_t, 'S': bvp1_S, 'V': bvp1_V},
        'bvp2': {'t': bvp2_t, 'S': bvp2_S, 'V': bvp2_V},
    }


class PINNLoss:
    """
    Physics-Informed Neural Network loss function for option pricing.
    
    Combines:
    - Data loss: MSE between predictions and market prices
    - Physics loss: MSE of Black-Scholes PDE residual
    - Boundary loss: MSE at boundary conditions (optional)
    """
    
    def __init__(self, r: float = 0.05, sigma: float = 0.2,
                 physics_weight: float = 1.0,
                 boundary_weight: float = 1.0):
        """
        Args:
            r: Risk-free interest rate
            sigma: Volatility (can use IV or historical vol)
            physics_weight: Weight for physics loss term
            boundary_weight: Weight for boundary condition loss
        """
        self.r = r
        self.sigma = sigma
        self.physics_weight = physics_weight
        self.boundary_weight = boundary_weight
        self.mse = tf.keras.losses.MeanSquaredError()
    
    def data_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Standard MSE loss on market data."""
        return self.mse(y_true, y_pred)
    
    def physics_loss(self, model: tf.keras.Model, 
                     t: tf.Tensor, S: tf.Tensor) -> tf.Tensor:
        """
        Compute physics loss from PDE residual.
        
        Args:
            model: Neural network (must accept [t, S] inputs)
            t: Time values
            S: Spot price values
            
        Returns:
            MSE of PDE residual (should be minimized to zero)
        """
        residual = compute_bs_residual(model, t, S, self.r, self.sigma)
        return tf.reduce_mean(tf.square(residual))
    
    def physics_loss_features(self, model: tf.keras.Model,
                              features: dict) -> tf.Tensor:
        """
        Compute physics loss using feature-space representation.
        
        Args:
            model: Neural network with 5+ input features
            features: Dictionary with maturity, k_over_s, r, iv, cond_vol
                      and optionally EPU, pos, neu, neg
                      Values can be numpy arrays or tensors
            
        Returns:
            MSE of approximate PDE residual
        """
        # Convert base features to tensors
        maturity = tf.cast(features['maturity'], tf.float32)
        k_over_s = tf.cast(features['k_over_s'], tf.float32)
        r_tensor = tf.cast(features['r'], tf.float32)
        iv = tf.cast(features['iv'], tf.float32)
        cond_vol = tf.cast(features['cond_vol'], tf.float32)
        
        # Check for extra features and combine them
        extra_feature_names = ['EPU', 'pos', 'neu', 'neg']
        extra_list = []
        for name in extra_feature_names:
            if name in features:
                extra_list.append(tf.cast(features[name], tf.float32))
        
        extra_features = None
        if extra_list:
            extra_features = tf.concat(extra_list, axis=1)
        
        residual = compute_bs_residual_from_features(
            model,
            maturity,
            k_over_s,
            r_tensor,
            iv,
            cond_vol,
            self.r,
            self.sigma,
            extra_features
        )
        return tf.reduce_mean(tf.square(residual))
    
    def total_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                   physics_loss: tf.Tensor) -> tf.Tensor:
        """
        Compute total PINN loss.
        
        Args:
            y_true: Ground truth values
            y_pred: Model predictions
            physics_loss: Pre-computed physics loss
            
        Returns:
            Weighted sum of data and physics losses
        """
        d_loss = self.data_loss(y_true, y_pred)
        return d_loss + self.physics_weight * physics_loss
