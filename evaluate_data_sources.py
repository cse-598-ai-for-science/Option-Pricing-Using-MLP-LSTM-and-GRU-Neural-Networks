"""
evaluate_data_sources.py

Fetch option-chain data from Yahoo Finance and train the same MLP on different
sources of data (baseline Yahoo-only, with EPU, with Sentiment, with both).
Produces bar charts for RMSE and MAE across data sources.

Usage (example):
  python evaluate_data_sources.py --ticker SPY --expirations 2 --epochs 50 --batch_size 32 \
    --epu_csv All_Daily_Policy_Data.csv --sentiment_csv sp500_news_290k_articles_cleaned.csv

By default this script will attempt to import TensorFlow; if TF is not installed
use --fetch-only to only fetch/prepare datasets without training.

"""

import argparse
import os
import logging
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse helpers from train_alt.py and data_loader_alt.py
from train_alt import build_dataset, prepare_xy
from train_alt import build_model  # supports model_type parameter
from data_loader_alt import load_epu_data, load_sentiment_data

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


def run_single_experiment(df: pd.DataFrame, epochs: int, batch_size: int, random_state: int = 42, 
                          verbose: int = 0, model_type: str = 'mlp', 
                          hidden_width: int = 64, n_layers: int = 4) -> Tuple[float, float]:
    """Train model on dataset and return (rmse_original_space, mae_original_space).

    We follow the same preprocessing as `train_alt.train_and_evaluate`: scale features,
    split train/test, train the MLP, predict on test, compute metrics in both transformed
    (log) space and original C/S space; return the latter for reporting.
    """
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow is not available in the environment. Install it or run with --fetch-only.')

    # prepare X, y using the same function (which applies dropna & log transform)
    X, y, feat_names = prepare_xy(df, use_log_target=True)

    # small safety check
    if X.shape[0] < 10:
        raise RuntimeError(f"Not enough samples to train (found {X.shape[0]}). Try increasing expirations or choose a more liquid ticker.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(X_train.shape[1], model_type=model_type, hidden_width=hidden_width,
                        n_layers=n_layers)

    # set reproducible seed for TF
    try:
        tf.random.set_seed(random_state)
    except Exception:
        pass

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.1)

    y_pred = model.predict(X_test).flatten()

    # metrics in transformed (log) space
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # convert back to original C/S space for more interpretable metrics
    try:
        y_test_orig = np.exp(y_test)
        y_pred_orig = np.exp(y_pred)
        rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    except Exception:
        # fallback: return metrics in transformed space
        rmse_orig, mae_orig = rmse, mae

    return float(rmse_orig), float(mae_orig)


def aggregate_experiments(results: Dict[str, list]) -> Dict[str, Tuple[float, float]]:
    """Compute mean RMSE/MAE per data source from multiple repeats."""
    out = {}
    for key, vals in results.items():
        arr = np.array(vals)
        # each val is (rmse, mae)
        if arr.size == 0:
            out[key] = (float('nan'), float('nan'))
            continue
        mean_rmse = float(np.nanmean(arr[:, 0]))
        mean_mae = float(np.nanmean(arr[:, 1]))
        out[key] = (mean_rmse, mean_mae)
    return out


def plot_metrics(metrics: Dict[str, Tuple[float, float]], out_dir: str):
    """Create comprehensive bar charts for RMSE and MAE and save them to out_dir.
    
    Features:
    - Rotated x-axis labels for readability
    - Different colors for different architectures (MLP vs Gated)
    - Grouped bar charts when comparing architectures
    - Multiple chart types for robust visualization
    """
    os.makedirs(out_dir, exist_ok=True)
    
    labels = list(metrics.keys())
    rmse_vals = [metrics[k][0] for k in labels]
    mae_vals = [metrics[k][1] for k in labels]
    
    # Define color scheme for architectures
    mlp_color = '#2E86AB'      # Blue for MLP
    gated_color = '#A23B72'    # Purple/Magenta for Gated
    default_color = '#28A745'  # Green for single architecture
    
    # Check if we have both architectures (labels contain _mlp and _gated)
    has_both = any('_mlp' in k for k in labels) and any('_gated' in k for k in labels)
    
    if has_both:
        # Create grouped bar charts
        _plot_grouped_metrics(metrics, out_dir, mlp_color, gated_color)
    else:
        # Create simple bar charts with rotated labels
        _plot_simple_metrics(labels, rmse_vals, mae_vals, out_dir, default_color)
    
    # Always create a combined comparison chart
    _plot_combined_chart(metrics, out_dir, mlp_color, gated_color, default_color)
    
    return (os.path.join(out_dir, 'rmse_by_data_source.png'),
            os.path.join(out_dir, 'mae_by_data_source.png'))


def _plot_simple_metrics(labels, rmse_vals, mae_vals, out_dir, color):
    """Create simple bar charts with rotated labels."""
    x = np.arange(len(labels))
    width = 0.6
    
    # RMSE Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, rmse_vals, width, color=color, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('RMSE (original C/S space)', fontsize=11)
    ax.set_title('RMSE by Data Source', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(rmse_vals) * 1.15)
    
    # Add value labels on bars
    for bar, val in zip(bars, rmse_vals):
        if not np.isnan(val):
            ax.annotate(f'{val:.5f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rmse_by_data_source.png'), dpi=150)
    plt.close()
    
    # MAE Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, mae_vals, width, color='#E8871E', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('MAE (original C/S space)', fontsize=11)
    ax.set_title('MAE by Data Source', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(mae_vals) * 1.15)
    
    for bar, val in zip(bars, mae_vals):
        if not np.isnan(val):
            ax.annotate(f'{val:.5f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mae_by_data_source.png'), dpi=150)
    plt.close()


def _plot_grouped_metrics(metrics: Dict[str, Tuple[float, float]], out_dir: str,
                          mlp_color: str, gated_color: str):
    """Create grouped bar charts comparing MLP and Gated architectures."""
    # Parse data sources and architectures
    data_sources = set()
    for key in metrics.keys():
        if '_mlp' in key:
            data_sources.add(key.replace('_mlp', ''))
        elif '_gated' in key:
            data_sources.add(key.replace('_gated', ''))
    
    data_sources = sorted(list(data_sources))
    x = np.arange(len(data_sources))
    width = 0.35
    
    mlp_rmse = [metrics.get(f'{ds}_mlp', (np.nan, np.nan))[0] for ds in data_sources]
    gated_rmse = [metrics.get(f'{ds}_gated', (np.nan, np.nan))[0] for ds in data_sources]
    mlp_mae = [metrics.get(f'{ds}_mlp', (np.nan, np.nan))[1] for ds in data_sources]
    gated_mae = [metrics.get(f'{ds}_gated', (np.nan, np.nan))[1] for ds in data_sources]
    
    # RMSE Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, mlp_rmse, width, label='MLP', color=mlp_color, 
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, gated_rmse, width, label='Gated', color=gated_color,
                   edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(data_sources, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('RMSE (original C/S space)', fontsize=11)
    ax.set_title('RMSE Comparison: MLP vs Gated Architecture', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    # Add value labels
    for bars, vals in [(bars1, mlp_rmse), (bars2, gated_rmse)]:
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.annotate(f'{val:.5f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=7, rotation=45)
    
    all_rmse = [v for v in mlp_rmse + gated_rmse if not np.isnan(v)]
    if all_rmse:
        ax.set_ylim(0, max(all_rmse) * 1.2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rmse_by_data_source.png'), dpi=150)
    plt.close()
    
    # MAE Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, mlp_mae, width, label='MLP', color=mlp_color,
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, gated_mae, width, label='Gated', color=gated_color,
                   edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(data_sources, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('MAE (original C/S space)', fontsize=11)
    ax.set_title('MAE Comparison: MLP vs Gated Architecture', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    for bars, vals in [(bars1, mlp_mae), (bars2, gated_mae)]:
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.annotate(f'{val:.5f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=7, rotation=45)
    
    all_mae = [v for v in mlp_mae + gated_mae if not np.isnan(v)]
    if all_mae:
        ax.set_ylim(0, max(all_mae) * 1.2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mae_by_data_source.png'), dpi=150)
    plt.close()
    
    # Create improvement percentage chart
    _plot_improvement_chart(data_sources, mlp_rmse, gated_rmse, out_dir)


def _plot_improvement_chart(data_sources, mlp_rmse, gated_rmse, out_dir):
    """Create a chart showing percentage improvement of Gated over MLP."""
    improvements = []
    valid_sources = []
    for i, ds in enumerate(data_sources):
        if not np.isnan(mlp_rmse[i]) and not np.isnan(gated_rmse[i]) and mlp_rmse[i] > 0:
            imp = ((mlp_rmse[i] - gated_rmse[i]) / mlp_rmse[i]) * 100
            improvements.append(imp)
            valid_sources.append(ds)
    
    if not improvements:
        return
    
    x = np.arange(len(valid_sources))
    colors = ['#28A745' if imp > 0 else '#DC3545' for imp in improvements]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, improvements, 0.6, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(valid_sources, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_title('Gated Architecture Improvement over MLP (RMSE)', fontsize=13, fontweight='bold')
    
    for bar, val in zip(bars, improvements):
        ypos = bar.get_height() + (1 if val >= 0 else -3)
        ax.annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, ypos),
                   ha='center', va='bottom' if val >= 0 else 'top', fontsize=9, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'improvement_chart.png'), dpi=150)
    plt.close()


def _plot_combined_chart(metrics: Dict[str, Tuple[float, float]], out_dir: str,
                         mlp_color: str, gated_color: str, default_color: str):
    """Create a combined metrics heatmap-style visualization."""
    labels = list(metrics.keys())
    rmse_vals = np.array([metrics[k][0] for k in labels])
    mae_vals = np.array([metrics[k][1] for k in labels])
    
    # Skip if all values are NaN
    if np.all(np.isnan(rmse_vals)) and np.all(np.isnan(mae_vals)):
        return
    
    # Create horizontal bar chart for better label readability
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(labels) * 0.5)))
    
    y = np.arange(len(labels))
    
    # Assign colors based on architecture
    colors = []
    for label in labels:
        if '_mlp' in label:
            colors.append(mlp_color)
        elif '_gated' in label:
            colors.append(gated_color)
        else:
            colors.append(default_color)
    
    # RMSE horizontal bars
    ax1 = axes[0]
    bars1 = ax1.barh(y, rmse_vals, 0.6, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel('RMSE (original C/S space)', fontsize=11)
    ax1.set_title('RMSE by Configuration', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    for bar, val in zip(bars1, rmse_vals):
        if not np.isnan(val):
            ax1.annotate(f'{val:.5f}', xy=(val, bar.get_y() + bar.get_height()/2),
                        ha='left', va='center', fontsize=8, xytext=(3, 0),
                        textcoords='offset points')
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # MAE horizontal bars
    ax2 = axes[1]
    bars2 = ax2.barh(y, mae_vals, 0.6, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel('MAE (original C/S space)', fontsize=11)
    ax2.set_title('MAE by Configuration', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    
    for bar, val in zip(bars2, mae_vals):
        if not np.isnan(val):
            ax2.annotate(f'{val:.5f}', xy=(val, bar.get_y() + bar.get_height()/2),
                        ha='left', va='center', fontsize=8, xytext=(3, 0),
                        textcoords='offset points')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add legend if multiple architectures
    has_mlp = any('_mlp' in k for k in labels)
    has_gated = any('_gated' in k for k in labels)
    if has_mlp and has_gated:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=mlp_color, edgecolor='black', label='MLP'),
            Patch(facecolor=gated_color, edgecolor='black', label='Gated')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
                   bbox_to_anchor=(0.5, 0.02), fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(os.path.join(out_dir, 'combined_metrics.png'), dpi=150)
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run experiments comparing data sources for option-pricing MLP')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker to fetch options for (default AAPL)')
    parser.add_argument('--expirations', type=int, default=2, help='Limit number of expirations to fetch (default 2)')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs (default 20)')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size (default 32)')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'gated', 'both'],
                        help='Model architecture: mlp, gated, or both for comparison (default mlp)')
    parser.add_argument('--hidden_width', type=int, default=64, help='Hidden layer width for gated model (default 64)')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of residual blocks for gated model (default 4)')
    parser.add_argument('--repeats', type=int, default=3, help='Number of repeats to average results (default 3)')
    parser.add_argument('--epu_csv', type=str, default=None, help='Path to EPU CSV (optional)')
    parser.add_argument('--sentiment_csv', type=str, default=None, help='Path to Sentiment CSV (optional)')
    parser.add_argument('--use_garch', action='store_true', help='Use GARCH for conditional vol (arch package required)')
    parser.add_argument('--fetch-only', action='store_true', help='Only fetch/prepare datasets; do not train (useful when TF missing)')
    parser.add_argument('--out_dir', type=str, default='evaluation_results', help='Output directory for plots (default: evaluation_results)')
    parser.add_argument('--min_samples', type=int, default=10, help='Minimum samples required to attempt training (default 10)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output during training and more console logging')

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('evaluate_data_sources')

    # Load optional external data
    epu_df = None
    if args.epu_csv:
        logger.info('Loading EPU CSV from %s', args.epu_csv)
        epu_df = load_epu_data(args.epu_csv)

    sentiment_df = None
    if args.sentiment_csv:
        logger.info('Loading Sentiment CSV from %s', args.sentiment_csv)
        sentiment_df = load_sentiment_data(args.sentiment_csv)

    # Build datasets for four configurations
    configs = {
        'yf_only': {'epu_df': None, 'sentiment_df': None},
        'epu': {'epu_df': epu_df, 'sentiment_df': None},
        'sentiment': {'epu_df': None, 'sentiment_df': sentiment_df},
        'both': {'epu_df': epu_df, 'sentiment_df': sentiment_df}
    }

    # Determine which model types to test
    if args.model == 'both':
        model_types = ['mlp', 'gated']
    else:
        model_types = [args.model]

    results = {}
    per_run_records = []

    # If TF is not available and not fetch-only, fail with helpful message
    if not TF_AVAILABLE and not args.fetch_only:
        logger.error('TensorFlow not available. Install it or run with --fetch-only to only fetch/prepare datasets.')
        sys.exit(1)

    for model_type in model_types:
        logger.info('Testing model type: %s', model_type)
        
        for cfg_name, cfg in configs.items():
            # Create combined key for results
            result_key = f"{cfg_name}_{model_type}" if len(model_types) > 1 else cfg_name
            results[result_key] = []
            
            logger.info('Running config: %s with model: %s', cfg_name, model_type)
            try:
                df = build_dataset(args.ticker, expirations_limit=args.expirations, risk_free_rate=0.0,
                                   use_garch=args.use_garch, epu_df=cfg['epu_df'], sentiment_df=cfg['sentiment_df'])
            except Exception as e:
                logger.warning('Failed to build dataset for %s: %s', cfg_name, e)
                per_run_records.append({'data_source': cfg_name, 'model_type': model_type, 'repeat': 0, 
                                        'sample_count': 0, 'rmse': float('nan'), 'mae': float('nan'), 
                                        'error': f'build_failed: {e}'})
                continue

            logger.info('Prepared dataset for %s: %d rows', cfg_name, len(df))

            # Quick check: compute sample count after prepare_xy
            sample_count = 0
            try:
                X_tmp, y_tmp, feat_tmp = prepare_xy(df, use_log_target=True)
                sample_count = X_tmp.shape[0]
                logger.info('Samples after prepare_xy for %s: %d (features: %s)', cfg_name, sample_count, feat_tmp)
            except Exception as e:
                logger.warning('prepare_xy failed for %s: %s', cfg_name, e)
                per_run_records.append({'data_source': cfg_name, 'model_type': model_type, 'repeat': 0, 
                                        'sample_count': 0, 'rmse': float('nan'), 'mae': float('nan'), 
                                        'error': f'prepare_xy_failed: {e}'})
                continue

            if args.fetch_only:
                results[result_key].append((float('nan'), float('nan')))
                per_run_records.append({'data_source': cfg_name, 'model_type': model_type, 'repeat': 0, 
                                        'sample_count': sample_count, 'rmse': float('nan'), 'mae': float('nan'), 
                                        'error': 'fetch_only'})
                continue

            if sample_count < args.min_samples:
                logger.warning('Skipping config %s: not enough samples (%d < %d)', cfg_name, sample_count, args.min_samples)
                per_run_records.append({'data_source': cfg_name, 'model_type': model_type, 'repeat': 0, 
                                        'sample_count': sample_count, 'rmse': float('nan'), 'mae': float('nan'), 
                                        'error': 'not_enough_samples'})
                continue

            # run multiple repeats with different seeds
            for rep in range(args.repeats):
                seed = 42 + rep
                try:
                    log_verb = 1 if args.verbose else 0
                    logger.info('Starting training for %s/%s run %d (seed=%d)', cfg_name, model_type, rep+1, seed)
                    rmse, mae = run_single_experiment(df, epochs=args.epochs, batch_size=args.batch_size, 
                                                      random_state=seed, verbose=log_verb,
                                                      model_type=model_type, hidden_width=args.hidden_width,
                                                      n_layers=args.n_layers)
                    logger.info('Result [%s/%s run %d]: RMSE=%0.6f MAE=%0.6f', cfg_name, model_type, rep + 1, rmse, mae)
                    results[result_key].append((rmse, mae))
                    per_run_records.append({'data_source': cfg_name, 'model_type': model_type, 'repeat': rep + 1, 
                                            'sample_count': sample_count, 'rmse': rmse, 'mae': mae, 'error': ''})
                except Exception as e:
                    logger.warning('Failed experiment for %s/%s (run %d): %s', cfg_name, model_type, rep + 1, e)
                    per_run_records.append({'data_source': cfg_name, 'model_type': model_type, 'repeat': rep + 1, 
                                            'sample_count': sample_count, 'rmse': float('nan'), 'mae': float('nan'), 
                                            'error': str(e)})

    agg = aggregate_experiments(results)
    logger.info('Aggregated results: %s', agg)

    # Save numeric results as CSV
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for k, (r, m) in agg.items():
        rows.append({'data_source': k, 'rmse': r, 'mae': m})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'aggregated_metrics.csv'), index=False)

    # Save per-run records for debugging and reproducibility
    try:
        per_run_df = pd.DataFrame(per_run_records)
        per_run_df.to_csv(os.path.join(out_dir, 'per_run_metrics.csv'), index=False)
        logger.info('Saved per-run metrics to %s', os.path.join(out_dir, 'per_run_metrics.csv'))
    except Exception as e:
        logger.warning('Failed to save per-run metrics: %s', e)

    # Plot and save figures (if at least one numeric metric exists)
    try:
        rmse_path, mae_path = plot_metrics(agg, out_dir)
        logger.info('Saved RMSE plot to %s and MAE plot to %s', rmse_path, mae_path)
    except Exception as e:
        logger.warning('Failed to create plots: %s', e)

    logger.info('Done. Results directory: %s', out_dir)


if __name__ == '__main__':
    main()
