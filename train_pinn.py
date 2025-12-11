"""
train_pinn.py

Physics-Informed Neural Network training for option pricing.
Incorporates Black-Scholes PDE residual into the loss function.

This script uses a custom training loop to compute:
- Data loss: MSE between predictions and market option prices
- Physics loss: MSE of Black-Scholes PDE residual

Usage:
  python train_pinn.py --ticker SPY --expirations 2 --epochs 100 --physics_weight 10.0

  # With alternative data
  python train_pinn.py --ticker SPY --expirations 2 --epochs 100 \
    --epu_csv All_Daily_Policy_Data.csv \
    --sentiment_csv sp500_news_290k_articles_cleaned.csv
"""

import argparse
import sys
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Data loading
from data_loader_alt import yfinance_to_model_df, load_epu_data, load_sentiment_data

# Lazy imports for TensorFlow
try:
    import tensorflow as tf
    from models.gated_mlp import build_gated_model
    from models.base_mlp import build_base_mlp
    from models.physics_loss import (
        PINNLoss, 
        sample_physics_features,
        compute_bs_residual_from_features
    )
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False


def plot_physics_metrics(out_dir: str):
    """Generate comprehensive charts for physics-informed training results.
    
    Creates charts from the aggregated_metrics.csv file showing:
    - RMSE by physics weight
    - Comparison across data configurations
    - Model performance heatmap
    """
    agg_path = os.path.join(out_dir, 'aggregated_metrics.csv')
    if not os.path.exists(agg_path):
        return
    
    try:
        df = pd.read_csv(agg_path)
    except Exception:
        return
    
    if df.empty:
        return
    
    # Define color scheme
    weight_colors = {
        0.0: '#2E86AB',    # Blue
        0.1: '#A23B72',    # Purple
        1.0: '#F18F01',    # Orange
        5.0: '#C73E1D',    # Red
        10.0: '#3B1F2B',   # Dark
    }
    default_color = '#28A745'  # Green
    
    # Chart 1: RMSE by Physics Weight
    _plot_rmse_by_physics_weight(df, out_dir, weight_colors, default_color)
    
    # Chart 2: RMSE by Configuration (grouped by physics weight)
    _plot_rmse_by_config(df, out_dir)
    
    # Chart 3: Physics Weight Sensitivity
    _plot_weight_sensitivity(df, out_dir)
    
    # Chart 4: Combined horizontal bar chart
    _plot_physics_combined(df, out_dir)


def _plot_rmse_by_physics_weight(df: pd.DataFrame, out_dir: str, 
                                  weight_colors: dict, default_color: str):
    """Plot RMSE by physics weight."""
    # Group by physics weight
    if 'physics_weight' not in df.columns or 'rmse_orig_mean' not in df.columns:
        return
    
    grouped = df.groupby('physics_weight')['rmse_orig_mean'].mean().reset_index()
    grouped = grouped.sort_values('physics_weight')
    
    weights = grouped['physics_weight'].values
    rmse_vals = grouped['rmse_orig_mean'].values
    
    colors = [weight_colors.get(w, default_color) for w in weights]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(weights))
    bars = ax.bar(x, rmse_vals, 0.6, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{w:.1f}' for w in weights], fontsize=11)
    ax.set_xlabel('Physics Weight', fontsize=11)
    ax.set_ylabel('Mean RMSE (original C/S space)', fontsize=11)
    ax.set_title('RMSE by Physics Loss Weight', fontsize=13, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, rmse_vals):
        if not np.isnan(val):
            ax.annotate(f'{val:.5f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rmse_by_physics_weight.png'), dpi=150)
    plt.close()


def _plot_rmse_by_config(df: pd.DataFrame, out_dir: str):
    """Plot RMSE by configuration, grouped by physics weight."""
    if 'config' not in df.columns or 'physics_weight' not in df.columns:
        return
    
    # Get unique configs and physics weights
    configs = df['config'].unique()
    weights = sorted(df['physics_weight'].unique())
    
    if len(weights) < 2:
        return
    
    # Color palette for weights
    cmap = plt.cm.get_cmap('viridis', len(weights))
    colors = [cmap(i) for i in range(len(weights))]
    
    x = np.arange(len(configs))
    width = 0.8 / len(weights)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, weight in enumerate(weights):
        weight_data = df[df['physics_weight'] == weight]
        rmse_vals = []
        for config in configs:
            config_data = weight_data[weight_data['config'] == config]
            if not config_data.empty:
                rmse_vals.append(config_data['rmse_orig_mean'].mean())
            else:
                rmse_vals.append(np.nan)
        
        offset = (i - len(weights)/2 + 0.5) * width
        bars = ax.bar(x + offset, rmse_vals, width, label=f'weight={weight:.1f}',
                      color=colors[i], edgecolor='black', linewidth=0.3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Mean RMSE (original C/S space)', fontsize=11)
    ax.set_title('RMSE by Configuration and Physics Weight', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rmse_by_config_weight.png'), dpi=150)
    plt.close()


def _plot_weight_sensitivity(df: pd.DataFrame, out_dir: str):
    """Plot how RMSE changes with physics weight (line chart)."""
    if 'physics_weight' not in df.columns:
        return
    
    configs = df['config'].unique()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color palette for configs
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#28A745', '#C73E1D']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, config in enumerate(configs):
        config_data = df[df['config'] == config].sort_values('physics_weight')
        if len(config_data) > 1:
            ax.plot(config_data['physics_weight'], config_data['rmse_orig_mean'],
                   marker=markers[i % len(markers)], color=colors[i % len(colors)],
                   label=config, linewidth=2, markersize=8)
    
    ax.set_xlabel('Physics Weight', fontsize=11)
    ax.set_ylabel('Mean RMSE (original C/S space)', fontsize=11)
    ax.set_title('Physics Weight Sensitivity Analysis', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'physics_weight_sensitivity.png'), dpi=150)
    plt.close()


def _plot_physics_combined(df: pd.DataFrame, out_dir: str):
    """Create a combined horizontal bar chart for all configurations."""
    if df.empty:
        return
    
    # Create a label combining config and physics weight
    df = df.copy()
    df['label'] = df.apply(
        lambda row: f"{row.get('config', 'unknown')} (w={row.get('physics_weight', 0):.1f})", 
        axis=1
    )
    
    labels = df['label'].values
    rmse_vals = df['rmse_orig_mean'].values if 'rmse_orig_mean' in df.columns else df.get('rmse_log_mean', [np.nan] * len(df))
    
    # Skip if all NaN
    if np.all(np.isnan(rmse_vals)):
        return
    
    # Assign colors based on physics weight
    weight_colors = {
        0.0: '#2E86AB',
        0.1: '#A23B72', 
        1.0: '#F18F01',
        5.0: '#C73E1D',
        10.0: '#3B1F2B',
    }
    colors = [weight_colors.get(w, '#28A745') for w in df.get('physics_weight', [0] * len(df))]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.4)))
    y = np.arange(len(labels))
    
    bars = ax.barh(y, rmse_vals, 0.6, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mean RMSE (original C/S space)', fontsize=11)
    ax.set_title('Physics-Informed Training Results', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, rmse_vals):
        if not np.isnan(val):
            ax.annotate(f'{val:.5f}', xy=(val, bar.get_y() + bar.get_height()/2),
                       ha='left', va='center', fontsize=8, xytext=(3, 0),
                       textcoords='offset points')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'physics_combined_results.png'), dpi=150)
    plt.close()


def build_dataset(ticker: str, expirations_limit: int = None, risk_free_rate: float = 0.0,
                  use_garch: bool = False, 
                  epu_df: pd.DataFrame = None,
                  sentiment_df: pd.DataFrame = None) -> pd.DataFrame:
    """Fetches option chains and concatenates them."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    exps = t.options
    if not exps:
        raise RuntimeError(f"No expirations found for ticker {ticker}")
    if expirations_limit is not None:
        exps = exps[:expirations_limit]

    dfs = []
    for e in exps:
        df = yfinance_to_model_df(ticker, e, risk_free_rate=risk_free_rate, 
                                  use_garch=use_garch, 
                                  epu_df=epu_df,
                                  sentiment_df=sentiment_df)
        if df is None or df.empty:
            continue
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No option data fetched for {ticker}")
    return pd.concat(dfs, ignore_index=True)


def prepare_xy(df: pd.DataFrame, use_log_target: bool = True) -> Tuple[np.ndarray, np.ndarray, list]:
    """Prepare features and target from dataframe."""
    # Base features
    features = ['r', 'K_over_S', 'Maturity', 'IV', 'cond_vol']
    
    # Dynamically add external features
    if 'EPU' in df.columns and df['EPU'].notna().any():
        features.append('EPU')
        
    for col in ['pos', 'neu', 'neg']:
        if col in df.columns and df[col].notna().any():
            features.append(col)

    df2 = df.copy()
    df2 = df2.dropna(subset=features + ['C_over_S'])

    X = df2[features].astype(float).values
    y_raw = df2['C_over_S'].astype(float).values

    mask = y_raw > 0
    X = X[mask]
    y_raw = y_raw[mask]

    if use_log_target:
        y = np.log(y_raw)
    else:
        y = y_raw

    return X, y, features


def build_model(input_dim: int, model_type: str = 'gated', 
                hidden_width: int = 64, n_layers: int = 4) -> tf.keras.Model:
    """Build model for PINN training."""
    if model_type == 'gated':
        return build_gated_model(input_dim, hidden_width=hidden_width, n_layers=n_layers)
    else:
        return build_base_mlp(input_dim)


@tf.function
def train_step(model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer,
               X_batch: tf.Tensor, y_batch: tf.Tensor,
               physics_features: dict, pinn_loss: PINNLoss,
               use_physics: bool = True) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Single training step with PINN loss.
    
    Args:
        model: Neural network model
        optimizer: Keras optimizer
        X_batch: Input features batch
        y_batch: Target values batch
        physics_features: Dictionary of physics sampling features
        pinn_loss: PINNLoss instance
        use_physics: Whether to include physics loss
        
    Returns:
        Tuple of (total_loss, data_loss, physics_loss)
    """
    with tf.GradientTape() as tape:
        # Data loss
        y_pred = model(X_batch, training=True)
        data_loss = pinn_loss.data_loss(y_batch, y_pred)
        
        # Physics loss (if enabled)
        if use_physics:
            physics_loss = pinn_loss.physics_loss_features(model, physics_features)
            total_loss = data_loss + pinn_loss.physics_weight * physics_loss
        else:
            physics_loss = tf.constant(0.0)
            total_loss = data_loss
    
    # Compute and apply gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss, data_loss, physics_loss


def train_pinn(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               feature_names: list = None,
               epochs: int = 100, batch_size: int = 32,
               model_type: str = 'gated', hidden_width: int = 64, n_layers: int = 4,
               learning_rate: float = 0.001,
               physics_weight: float = 1.0,
               n_physics_samples: int = 1000,
               r: float = 0.05, sigma: float = 0.2,
               verbose: int = 1) -> Tuple[tf.keras.Model, dict]:
    """
    Train a PINN model for option pricing.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_names: List of feature names (to identify extra features for physics sampling)
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_type: 'mlp' or 'gated'
        hidden_width: Hidden layer width (gated model)
        n_layers: Number of residual blocks (gated model)
        learning_rate: Learning rate for Adam optimizer
        physics_weight: Weight for physics loss term
        n_physics_samples: Number of physics sampling points per batch
        r: Risk-free rate for Black-Scholes PDE
        sigma: Volatility for Black-Scholes PDE
        verbose: Verbosity level (0=silent, 1=progress)
        
    Returns:
        Tuple of (trained_model, history_dict)
    """
    # Determine extra features beyond the base 5
    base_features = {'r', 'K_over_S', 'Maturity', 'IV', 'cond_vol'}
    extra_features = None
    if feature_names:
        extra = [f for f in feature_names if f not in base_features]
        if extra:
            extra_features = extra
    input_dim = X_train.shape[1]
    
    # Build model
    model = build_model(input_dim, model_type=model_type, 
                        hidden_width=hidden_width, n_layers=n_layers)
    
    # Setup optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Setup PINN loss
    pinn_loss = PINNLoss(r=r, sigma=sigma, physics_weight=physics_weight)
    
    # Convert to tensors
    X_train_t = tf.constant(X_train, dtype=tf.float32)
    y_train_t = tf.constant(y_train.reshape(-1, 1), dtype=tf.float32)
    X_val_t = tf.constant(X_val, dtype=tf.float32)
    y_val_t = tf.constant(y_val.reshape(-1, 1), dtype=tf.float32)
    
    # Training history
    history = {
        'total_loss': [],
        'data_loss': [],
        'physics_loss': [],
        'val_loss': []
    }
    
    n_samples = X_train.shape[0]
    n_batches = max(1, n_samples // batch_size)
    
    for epoch in range(epochs):
        epoch_total_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_train_shuffled = tf.gather(X_train_t, indices)
        y_train_shuffled = tf.gather(y_train_t, indices)
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # Sample physics points for this batch (include extra features if needed)
            physics_features = sample_physics_features(n_physics_samples, 
                                                       extra_features=extra_features)
            # Convert to tensors
            physics_features = {k: tf.constant(v, dtype=tf.float32) 
                               for k, v in physics_features.items()}
            
            # Training step
            total_loss, data_loss, physics_loss = train_step(
                model, optimizer, X_batch, y_batch,
                physics_features, pinn_loss,
                use_physics=(physics_weight > 0)
            )
            
            epoch_total_loss += total_loss.numpy()
            epoch_data_loss += data_loss.numpy()
            epoch_physics_loss += physics_loss.numpy()
        
        # Average losses
        epoch_total_loss /= n_batches
        epoch_data_loss /= n_batches
        epoch_physics_loss /= n_batches
        
        # Validation loss
        val_pred = model(X_val_t, training=False)
        val_loss = tf.reduce_mean(tf.square(val_pred - y_val_t)).numpy()
        
        # Record history
        history['total_loss'].append(epoch_total_loss)
        history['data_loss'].append(epoch_data_loss)
        history['physics_loss'].append(epoch_physics_loss)
        history['val_loss'].append(val_loss)
        
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"loss: {epoch_total_loss:.6f} - "
                  f"data_loss: {epoch_data_loss:.6f} - "
                  f"physics_loss: {epoch_physics_loss:.6f} - "
                  f"val_loss: {val_loss:.6f}")
    
    return model, history


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray,
                   use_log_target: bool = True) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Metrics in transformed space
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'rmse_log': rmse,
        'mae_log': mae,
        'r2': r2
    }
    
    # Metrics in original space (if log transformed)
    if use_log_target:
        y_test_orig = np.exp(y_test)
        y_pred_orig = np.exp(y_pred)
        metrics['rmse_orig'] = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        metrics['mae_orig'] = mean_absolute_error(y_test_orig, y_pred_orig)
    
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Train Physics-Informed Neural Network for option pricing'
    )
    
    # Data arguments
    parser.add_argument('--ticker', default='AAPL', help='Ticker symbol')
    parser.add_argument('--expirations', type=int, default=None, 
                        help='Limit number of expirations')
    parser.add_argument('--risk_free_rate', type=float, default=0.05,
                        help='Risk-free rate for Black-Scholes PDE')
    parser.add_argument('--use_garch', action='store_true',
                        help='Use GARCH for conditional volatility')
    
    # External data
    parser.add_argument('--epu_csv', type=str, default=None,
                        help='Path to EPU CSV file')
    parser.add_argument('--sentiment_csv', type=str, default=None,
                        help='Path to Sentiment CSV file')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='gated', 
                        choices=['mlp', 'gated'],
                        help='Model architecture')
    parser.add_argument('--hidden_width', type=int, default=64,
                        help='Hidden layer width (gated model)')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of residual blocks (gated model)')
    
    # Physics arguments
    parser.add_argument('--physics_weight', type=float, default=1.0,
                        help='Weight for physics loss (0 to disable)')
    parser.add_argument('--n_physics_samples', type=int, default=1000,
                        help='Physics sampling points per batch')
    parser.add_argument('--sigma', type=float, default=0.2,
                        help='Volatility for Black-Scholes PDE')
    
    # Output arguments
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--save_model', type=str, default=None,
                        help='Path to save trained model')
    parser.add_argument('--fetch-only', action='store_true',
                        help='Only fetch data, skip training')
    parser.add_argument('--out_dir', type=str, default='evaluation_results/physics-informed-training',
                        help='Output directory for results CSV')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Number of experiment repeats with different seeds')
    
    args = parser.parse_args(argv)
    
    # Setup logging
    logging.basicConfig(
        filename='train_pinn.log',
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    logger = logging.getLogger('train_pinn')
    
    if args.verbose:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        logger.addHandler(console)
    
    # Check TensorFlow
    if not TF_AVAILABLE and not args.fetch_only:
        print('TensorFlow not available. Install it or use --fetch-only.')
        sys.exit(1)
    
    # Load external data
    epu_df = None
    if args.epu_csv:
        logger.info(f"Loading EPU data from {args.epu_csv}")
        epu_df = load_epu_data(args.epu_csv)
        logger.info(f"Loaded {len(epu_df)} EPU records")
    
    sentiment_df = None
    if args.sentiment_csv:
        logger.info(f"Loading Sentiment data from {args.sentiment_csv}")
        sentiment_df = load_sentiment_data(args.sentiment_csv)
        logger.info(f"Loaded {len(sentiment_df)} sentiment records")
    
    # Build dataset
    try:
        logger.info(f"Fetching option data for {args.ticker}")
        df = build_dataset(
            args.ticker,
            expirations_limit=args.expirations,
            risk_free_rate=args.risk_free_rate,
            use_garch=args.use_garch,
            epu_df=epu_df,
            sentiment_df=sentiment_df
        )
        logger.info(f"Fetched {len(df)} option rows")
    except Exception as e:
        logger.exception("Failed to build dataset")
        sys.exit(2)
    
    # Prepare features
    X, y, features = prepare_xy(df, use_log_target=True)
    logger.info(f"Prepared {X.shape[0]} samples with features: {features}")
    
    if args.fetch_only:
        logger.info("Fetch-only mode: skipping training")
        return
    
    # Determine config name for results
    config_name = 'yf_only'
    if epu_df is not None and sentiment_df is not None:
        config_name = 'both'
    elif epu_df is not None:
        config_name = 'epu'
    elif sentiment_df is not None:
        config_name = 'sentiment'
    
    # Results storage
    per_run_records = []
    all_metrics = []
    
    # Run experiments with multiple seeds
    for rep in range(args.repeats):
        seed = 42 + rep
        logger.info(f"Starting run {rep + 1}/{args.repeats} with seed={seed}")
        
        # Set seeds for reproducibility
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Train/test split with current seed
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        
        # Further split train into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=seed
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Training PINN model with physics_weight={args.physics_weight}")
        logger.info(f"Model: {args.model}, hidden_width={args.hidden_width}, n_layers={args.n_layers}")
        
        try:
            # Train model
            model, history = train_pinn(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                feature_names=features,
                epochs=args.epochs,
                batch_size=args.batch_size,
                model_type=args.model,
                hidden_width=args.hidden_width,
                n_layers=args.n_layers,
                learning_rate=args.learning_rate,
                physics_weight=args.physics_weight,
                n_physics_samples=args.n_physics_samples,
                r=args.risk_free_rate,
                sigma=args.sigma,
                verbose=1 if args.verbose else 0
            )
            
            # Evaluate
            metrics = evaluate_model(model, X_test_scaled, y_test, use_log_target=True)
            all_metrics.append(metrics)
            
            # Record per-run results
            per_run_records.append({
                'config': config_name,
                'model_type': args.model,
                'physics_weight': args.physics_weight,
                'n_physics_samples': args.n_physics_samples,
                'repeat': rep + 1,
                'seed': seed,
                'sample_count': X.shape[0],
                'feature_count': len(features),
                'rmse_log': metrics['rmse_log'],
                'mae_log': metrics['mae_log'],
                'r2': metrics['r2'],
                'rmse_orig': metrics.get('rmse_orig', float('nan')),
                'mae_orig': metrics.get('mae_orig', float('nan')),
                'error': ''
            })
            
            logger.info(f"Run {rep + 1} complete: RMSE={metrics['rmse_orig']:.6f}, R2={metrics['r2']:.4f}")
            
        except Exception as e:
            logger.exception(f"Run {rep + 1} failed: {e}")
            per_run_records.append({
                'config': config_name,
                'model_type': args.model,
                'physics_weight': args.physics_weight,
                'n_physics_samples': args.n_physics_samples,
                'repeat': rep + 1,
                'seed': seed,
                'sample_count': X.shape[0],
                'feature_count': len(features),
                'rmse_log': float('nan'),
                'mae_log': float('nan'),
                'r2': float('nan'),
                'rmse_orig': float('nan'),
                'mae_orig': float('nan'),
                'error': str(e)
            })
    
    # Aggregate metrics
    if all_metrics:
        avg_rmse_log = np.mean([m['rmse_log'] for m in all_metrics])
        avg_mae_log = np.mean([m['mae_log'] for m in all_metrics])
        avg_r2 = np.mean([m['r2'] for m in all_metrics])
        avg_rmse_orig = np.mean([m.get('rmse_orig', float('nan')) for m in all_metrics])
        avg_mae_orig = np.mean([m.get('mae_orig', float('nan')) for m in all_metrics])
        
        print("\n" + "="*60)
        print("AGGREGATED RESULTS")
        print("="*60)
        print(f"  Config: {config_name}")
        print(f"  Model: {args.model}")
        print(f"  Physics Weight: {args.physics_weight}")
        print(f"  Runs: {len(all_metrics)}/{args.repeats}")
        print(f"  Samples: {X.shape[0]}, Features: {len(features)}")
        print(f"  Avg RMSE (log space): {avg_rmse_log:.6f}")
        print(f"  Avg MAE (log space): {avg_mae_log:.6f}")
        print(f"  Avg R-squared: {avg_r2:.4f}")
        print(f"  Avg RMSE (original C/S): {avg_rmse_orig:.6f}")
        print(f"  Avg MAE (original C/S): {avg_mae_orig:.6f}")
        print("="*60)
    
    # Save results to CSV
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Per-run metrics
    per_run_df = pd.DataFrame(per_run_records)
    per_run_path = os.path.join(args.out_dir, 'per_run_metrics.csv')
    
    # Append to existing file if it exists, otherwise create new
    if os.path.exists(per_run_path):
        existing_df = pd.read_csv(per_run_path)
        per_run_df = pd.concat([existing_df, per_run_df], ignore_index=True)
    
    per_run_df.to_csv(per_run_path, index=False)
    logger.info(f"Saved per-run metrics to {per_run_path}")
    
    # Aggregated metrics
    if all_metrics:
        agg_record = {
            'config': config_name,
            'model_type': args.model,
            'physics_weight': args.physics_weight,
            'n_physics_samples': args.n_physics_samples,
            'runs': len(all_metrics),
            'sample_count': X.shape[0],
            'feature_count': len(features),
            'rmse_log_mean': avg_rmse_log,
            'mae_log_mean': avg_mae_log,
            'r2_mean': avg_r2,
            'rmse_orig_mean': avg_rmse_orig,
            'mae_orig_mean': avg_mae_orig
        }
        
        agg_path = os.path.join(args.out_dir, 'aggregated_metrics.csv')
        if os.path.exists(agg_path):
            existing_agg = pd.read_csv(agg_path)
            agg_df = pd.concat([existing_agg, pd.DataFrame([agg_record])], ignore_index=True)
        else:
            agg_df = pd.DataFrame([agg_record])
        
        agg_df.to_csv(agg_path, index=False)
        logger.info(f"Saved aggregated metrics to {agg_path}")
    
    # Generate charts from aggregated results
    try:
        plot_physics_metrics(args.out_dir)
        logger.info(f"Generated charts in {args.out_dir}")
    except Exception as e:
        logger.warning(f"Failed to generate charts: {e}")
    
    # Save model if requested (only last run)
    if args.save_model and 'model' in dir():
        model.save(args.save_model)
        logger.info(f"Model saved to {args.save_model}")
    
    logger.info("Training completed successfully")
    
    logger.info("Training completed successfully")


if __name__ == '__main__':
    main()
