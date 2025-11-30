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
from train_alt import build_model as _build_model  # reuse the same MLP
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


def run_single_experiment(df: pd.DataFrame, epochs: int, batch_size: int, random_state: int = 42, verbose: int = 0) -> Tuple[float, float]:
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

    model = _build_model(X_train.shape[1])

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
    """Create bar charts for RMSE and MAE and save them to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    labels = list(metrics.keys())
    rmse_vals = [metrics[k][0] for k in labels]
    mae_vals = [metrics[k][1] for k in labels]

    x = np.arange(len(labels))
    width = 0.6

    plt.figure(figsize=(8, 5))
    plt.bar(x, rmse_vals, width, color='tab:blue')
    plt.xticks(x, labels)
    plt.ylabel('RMSE (original C/S space)')
    plt.title('RMSE by data source')
    plt.tight_layout()
    rmse_path = os.path.join(out_dir, 'rmse_by_data_source.png')
    plt.savefig(rmse_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(x, mae_vals, width, color='tab:orange')
    plt.xticks(x, labels)
    plt.ylabel('MAE (original C/S space)')
    plt.title('MAE by data source')
    plt.tight_layout()
    mae_path = os.path.join(out_dir, 'mae_by_data_source.png')
    plt.savefig(mae_path)
    plt.close()

    return rmse_path, mae_path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run experiments comparing data sources for option-pricing MLP')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker to fetch options for (default AAPL)')
    parser.add_argument('--expirations', type=int, default=2, help='Limit number of expirations to fetch (default 2)')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs (default 20)')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size (default 32)')
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

    results = {k: [] for k in configs.keys()}
    per_run_records = []

    # If TF is not available and not fetch-only, fail with helpful message
    if not TF_AVAILABLE and not args.fetch_only:
        logger.error('TensorFlow not available. Install it or run with --fetch-only to only fetch/prepare datasets.')
        sys.exit(1)

    for cfg_name, cfg in configs.items():
        logger.info('Running config: %s', cfg_name)
        try:
            df = build_dataset(args.ticker, expirations_limit=args.expirations, risk_free_rate=0.0,
                               use_garch=args.use_garch, epu_df=cfg['epu_df'], sentiment_df=cfg['sentiment_df'])
        except Exception as e:
            logger.warning('Failed to build dataset for %s: %s', cfg_name, e)
            # record failure for diagnostics
            per_run_records.append({'data_source': cfg_name, 'repeat': 0, 'sample_count': 0, 'rmse': float('nan'), 'mae': float('nan'), 'error': f'build_failed: {e}'})
            continue

        logger.info('Prepared dataset for %s: %d rows', cfg_name, len(df))

        # Quick check: compute sample count after prepare_xy so we can decide whether to attempt training
        sample_count = 0
        try:
            X_tmp, y_tmp, feat_tmp = prepare_xy(df, use_log_target=True)
            sample_count = X_tmp.shape[0]
            logger.info('Samples after prepare_xy for %s: %d (features: %s)', cfg_name, sample_count, feat_tmp)
        except Exception as e:
            logger.warning('prepare_xy failed for %s: %s', cfg_name, e)
            per_run_records.append({'data_source': cfg_name, 'repeat': 0, 'sample_count': 0, 'rmse': float('nan'), 'mae': float('nan'), 'error': f'prepare_xy_failed: {e}'})
            continue

        if args.fetch_only:
            # store dataset size as NaN metrics (so they show up in plots as missing)
            results[cfg_name].append((float('nan'), float('nan')))
            per_run_records.append({'data_source': cfg_name, 'repeat': 0, 'sample_count': sample_count, 'rmse': float('nan'), 'mae': float('nan'), 'error': 'fetch_only'})
            continue

        if sample_count < args.min_samples:
            logger.warning('Skipping config %s: not enough samples after cleaning (%d < %d)', cfg_name, sample_count, args.min_samples)
            per_run_records.append({'data_source': cfg_name, 'repeat': 0, 'sample_count': sample_count, 'rmse': float('nan'), 'mae': float('nan'), 'error': 'not_enough_samples'})
            continue

        # run multiple repeats with different seeds for averaging
        for rep in range(args.repeats):
            seed = 42 + rep
            try:
                log_verb = 1 if args.verbose else 0
                logger.info('Starting training for %s run %d (seed=%d)', cfg_name, rep+1, seed)
                rmse, mae = run_single_experiment(df, epochs=args.epochs, batch_size=args.batch_size, random_state=seed, verbose=log_verb)
                logger.info('Result [%s run %d]: RMSE=%0.6f MAE=%0.6f', cfg_name, rep + 1, rmse, mae)
                results[cfg_name].append((rmse, mae))
                per_run_records.append({'data_source': cfg_name, 'repeat': rep + 1, 'sample_count': sample_count, 'rmse': rmse, 'mae': mae, 'error': ''})
            except Exception as e:
                logger.warning('Failed experiment for %s (run %d): %s', cfg_name, rep + 1, e)
                per_run_records.append({'data_source': cfg_name, 'repeat': rep + 1, 'sample_count': sample_count, 'rmse': float('nan'), 'mae': float('nan'), 'error': str(e)})

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
