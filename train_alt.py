"""
train_from_yf.py

Pulls option data from Yahoo Finance, incorporates EPU and Sentiment indices,
builds a model dataset, and trains a simple MLP.
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
from data_loader_alt import yfinance_to_model_df, load_epu_data, load_sentiment_data

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LeakyReLU, ELU
except Exception as e:
    tf = None


def build_dataset(ticker: str, expirations_limit: int = None, risk_free_rate: float = 0.0,
                  use_garch: bool = False, 
                  epu_df: pd.DataFrame = None,
                  sentiment_df: pd.DataFrame = None) -> pd.DataFrame:
    """Fetches option chains and concatenates them, merging EPU/Sentiment if provided."""
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
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def prepare_xy(df: pd.DataFrame, use_log_target: bool = True) -> Tuple[np.ndarray, np.ndarray, list]:
    """Selects features and target."""
    
    # Base features
    features = ['r', 'K_over_S', 'Maturity', 'IV', 'cond_vol']
    
    # Dynamically add external features if they exist in the df
    if 'EPU' in df.columns:
        features.append('EPU')
        print("-> Included feature: EPU")
        
    for col in ['pos', 'neu', 'neg']:
        if col in df.columns:
            features.append(col)
            print(f"-> Included feature: {col}")

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns in dataframe: {missing}")

    df2 = df.copy()
    # Drop rows where any feature or the target is NaN
    # This effectively drops rows where we couldn't find matching Sentiment/EPU dates
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


def build_model(input_dim: int) -> 'tf.keras.Model':
    """Builds a small MLP."""
    model = Sequential()
    model.add(Dense(30, input_dim=input_dim))
    model.add(LeakyReLU())
    model.add(Dense(60))
    model.add(ELU())
    model.add(Dense(90))
    model.add(LeakyReLU())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_and_evaluate(X: np.ndarray, y: np.ndarray, epochs: int = 20, batch_size: int = 32,
                       test_size: float = 0.2, random_state: int = 42, verbose: int = 1):
    """Scale features, train model, and print evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(X_train.shape[1])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                        validation_split=0.1)

    y_pred = model.predict(X_test).flatten()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('\nEvaluation (on held-out test set):')
    print(f'  Samples: {X.shape[0]}  Features: {X.shape[1]}')
    print(f'  RMSE (on transformed target): {rmse:.6f} | MAE: {mae:.6f} | R2: {r2:.4f}')

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
    parser.add_argument('--ticker', default='AAPL', help='Ticker symbol (e.g., AAPL).')
    parser.add_argument('--expirations', type=int, default=None, help='Limit number of expirations to fetch')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--risk_free_rate', type=float, default=0.0, help='Annual risk-free rate decimal')
    parser.add_argument('--use_garch', action='store_true', help='Compute conditional vol with GARCH')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose progress output')
    parser.add_argument('--fetch-only', action='store_true', help='Only fetch and prepare data')
    
    # External Data Arguments
    parser.add_argument('--epu_csv', type=str, default=None, help='Path to EPU index CSV file')
    parser.add_argument('--sentiment_csv', type=str, default=None, help='Path to Sentiment CSV file')
    
    args = parser.parse_args(argv)

    logging.basicConfig(filename='train_from_yf.log', level=logging.INFO,
                        filemode='w', format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger('train_from_yf')
    
    if args.verbose:
        import platform
        console_h = logging.StreamHandler(_sys.stdout)
        console_h.setLevel(logging.DEBUG)
        console_h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
        logger.addHandler(console_h)

    # 1. Load External Data
    epu_df = None
    if args.epu_csv:
        try:
            logger.info(f"Loading EPU data from {args.epu_csv}...")
            epu_df = load_epu_data(args.epu_csv)
            logger.info(f"Loaded {len(epu_df)} EPU records.")
        except Exception as e:
            logger.error(f"Failed to load EPU csv: {e}")
            sys.exit(1)

    sentiment_df = None
    if args.sentiment_csv:
        try:
            logger.info(f"Loading Sentiment data from {args.sentiment_csv}...")
            sentiment_df = load_sentiment_data(args.sentiment_csv)
            logger.info(f"Loaded {len(sentiment_df)} daily sentiment records.")
        except Exception as e:
            logger.error(f"Failed to load Sentiment csv: {e}")
            sys.exit(1)

    try:
        logger.info('Fetching option data for %s...', args.ticker)
        # 2. Pass both dataframes to build_dataset
        df = build_dataset(args.ticker, expirations_limit=args.expirations, risk_free_rate=args.risk_free_rate,
                           use_garch=args.use_garch, 
                           epu_df=epu_df, 
                           sentiment_df=sentiment_df)
    except Exception as e:
        logger.exception('Error while fetching/building dataset:')
        sys.exit(2)
    
    logger.info('Fetched %d option rows (before cleaning).', len(df))

    # 3. Prepare X, y (automatically detects new columns)
    X, y, feat_names = prepare_xy(df, use_log_target=True)
    logger.info('Prepared dataset: %d samples, %d features -> %s', X.shape[0], len(feat_names), feat_names)

    if not args.fetch_only and tf is None:
        print('TensorFlow not available. Run with --fetch-only to skip training.')
        sys.exit(1)

    if args.fetch_only:
        logger.info('Fetch-only mode: prepared dataset; skipping model training.')
        if args.verbose:
            logger.info('\nDataframe head:\n%s', df.head().to_string())
        return

    try:
        model, scaler, history = train_and_evaluate(X, y, epochs=args.epochs, batch_size=args.batch_size)
    except Exception:
        logger.exception('Error during training:')
        sys.exit(3)

    logger.info('Training completed successfully.')


if __name__ == '__main__':
    main()