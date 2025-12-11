"""
train_from_yf.py

Pulls option data from Yahoo Finance using helpers in `data_loader.py`, builds a model dataset,
trains a simple MLP similar to the notebook, and prints evaluation metrics.

Usage (simple):
  python train_from_yf.py --ticker AAPL --expirations 3 --epochs 5

The script expects `data_loader.py` to exist in the same directory (it uses
`yfinance_to_model_df` exposed there).
"""

import argparse
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import logging
import sys as _sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# relative import from local file
from data_loader import yfinance_to_model_df

# Lazy import of tensorflow so the script fails with a helpful message if missing
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LeakyReLU, ELU
    from models.gated_mlp import build_gated_model
    from models.base_mlp import build_base_mlp
except Exception as e:
    tf = None
    build_gated_model = None
    build_base_mlp = None


def build_dataset(ticker: str, expirations_limit: int = None, risk_free_rate: float = 0.0,
                  use_garch: bool = False) -> pd.DataFrame:
    """Fetches option chains for the given ticker and all (or limited) expirations and
    concatenates them into a single DataFrame using `yfinance_to_model_df` from `data_loader.py`.
    """
    import yfinance as yf
    t = yf.Ticker(ticker)
    exps = t.options
    if not exps:
        raise RuntimeError(f"No expirations found for ticker {ticker}")
    if expirations_limit is not None:
        exps = exps[:expirations_limit]

    dfs = []
    for e in exps:
        df = yfinance_to_model_df(ticker, e, risk_free_rate=risk_free_rate, use_garch=use_garch)
        if df is None or df.empty:
            continue
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No option data fetched for {ticker} (checked {len(exps)} expirations)")
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def prepare_xy(df: pd.DataFrame, use_log_target: bool = True) -> Tuple[np.ndarray, np.ndarray, list]:
    """Selects features, cleans rows with missing values and returns X, y, and feature names.

    Defaults mimic the notebook: features = ['r', 'K_over_S', 'Maturity', 'IV', 'cond_vol']
    Target is C_over_S (option price divided by underlying price). When use_log_target=True we
    take the natural log of positive C_over_S values and drop non-positive prices.
    """
    features = ['r', 'K_over_S', 'Maturity', 'IV', 'cond_vol']
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns in dataframe: {missing}")

    df2 = df.copy()
    # drop rows with NaN in features or target
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


def build_model(input_dim: int, model_type: str = 'mlp', hidden_width: int = 64,
                n_layers: int = 4) -> 'tf.keras.Model':
    """
    Builds a neural network model for option pricing.
    
    Args:
        input_dim: Number of input features
        model_type: 'mlp' for baseline or 'gated' for gated residual architecture
        hidden_width: Width of hidden layers (gated model only)
        n_layers: Number of residual blocks (gated model only)
        
    Returns:
        Compiled Keras model
    """
    if model_type == 'gated':
        return build_gated_model(input_dim, hidden_width=hidden_width, n_layers=n_layers)
    else:
        return build_base_mlp(input_dim)


def train_and_evaluate(X: np.ndarray, y: np.ndarray, epochs: int = 20, batch_size: int = 32,
                       test_size: float = 0.2, random_state: int = 42, verbose: int = 1,
                       model_type: str = 'mlp', hidden_width: int = 64, n_layers: int = 4):
    """Scale features, train model, and print evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(X_train.shape[1], model_type=model_type, hidden_width=hidden_width,
                        n_layers=n_layers)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                        validation_split=0.1)

    y_pred = model.predict(X_test).flatten()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('\nEvaluation (on held-out test set):')
    print(f'  Samples: {X.shape[0]}  Features: {X.shape[1]}')
    print(f'  RMSE (on transformed target): {rmse:.6f} | MAE: {mae:.6f} | R2: {r2:.4f}')

    # If target was log, also print metrics in original price-space for interpretability
    try:
        y_test_orig = np.exp(y_test)
        y_pred_orig = np.exp(y_pred)
        rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
        print(f'  RMSE (original C/S space): {rmse_orig:.6f} | MAE: {mae_orig:.6f}')
    except Exception:
        pass

    return model, scaler, history


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train option pricing MLP using data from Yahoo Finance')
    parser.add_argument('--ticker', default='AAPL', help='Ticker symbol (e.g., AAPL). Defaults to AAPL if not provided.')
    parser.add_argument('--expirations', type=int, default=None, help='Limit number of expirations to fetch')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'gated'],
                        help='Model architecture: mlp (baseline) or gated (residual)')
    parser.add_argument('--hidden_width', type=int, default=64, help='Hidden layer width (gated model)')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of residual blocks (gated model)')
    parser.add_argument('--risk_free_rate', type=float, default=0.0, help='Annual risk-free rate decimal')
    parser.add_argument('--use_garch', action='store_true', help='Compute conditional vol with GARCH (arch package)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose progress output')
    parser.add_argument('--fetch-only', action='store_true', help='Only fetch and prepare data; do not train the model (does not require TensorFlow)')
    args = parser.parse_args(argv)

    # configure file logging for all runs (overwrite previous log)
    logging.basicConfig(filename='train_from_yf.log', level=logging.INFO,
                        filemode='w', format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger('train_from_yf')
    # when verbose, also log to console
    if args.verbose:
        import platform
        console_h = logging.StreamHandler(_sys.stdout)
        console_h.setLevel(logging.DEBUG)
        console_h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
        logger.addHandler(console_h)
        logger.info('Python executable: %s', _sys.executable)
        logger.info('Python version: %s', platform.python_version())
        logger.info('TensorFlow available: %s', 'yes' if tf is not None else 'no')

    try:
        if argv is None and args.ticker == 'AAPL':
            logger.info("No --ticker provided; defaulting to 'AAPL'. To use another ticker, pass --ticker TICKER.")
        logger.info('Fetching option data for %s...', args.ticker)
        df = build_dataset(args.ticker, expirations_limit=args.expirations, risk_free_rate=args.risk_free_rate,
                           use_garch=args.use_garch,)
    except Exception as e:
        import traceback
        logger.exception('Error while fetching/building dataset:')
        sys.exit(2)
    logger.info('Fetched %d option rows (before cleaning).', len(df))

    X, y, feat_names = prepare_xy(df, use_log_target=True)
    logger.info('Prepared dataset: %d samples, %d features -> %s', X.shape[0], len(feat_names), feat_names)

    # If not in fetch-only mode, ensure TensorFlow is available for training
    if not args.fetch_only and tf is None:
        print('TensorFlow (or Keras) not available. Install dependencies from requirements.txt to train the model.')
        print('You can run with --fetch-only to only fetch and prepare data without training.')
        sys.exit(1)

    if args.fetch_only:
        logger.info('Fetch-only mode: prepared dataset; skipping model training.')
        # Optionally show a few rows if verbose
        if args.verbose:
            logger.info('\nDataframe head:\n%s', df.head().to_string())
        return

    try:
        model, scaler, history = train_and_evaluate(X, y, epochs=args.epochs, batch_size=args.batch_size,
                                                    model_type=args.model, hidden_width=args.hidden_width,
                                                    n_layers=args.n_layers)
    except Exception:
        import traceback
        logger.exception('Error during training:')
        sys.exit(3)

    logger.info('Training completed successfully.')


if __name__ == '__main__':
    main()
