# data_loader_yf.py
import pandas as pd
import numpy as np
import yfinance as yf

try:
    from arch import arch_model
    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False

def _to_years(days: int):
    return days / 252.0

def load_epu_data(csv_path: str) -> pd.DataFrame:
    """
    Loads EPU data from CSV, combines day/month/year into a datetime index.
    Returns columns: ['Date', 'EPU']
    """
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df[['Date', 'daily_policy_index']].rename(columns={'daily_policy_index': 'EPU'})
    df = df.sort_values('Date')
    return df

def load_sentiment_data(csv_path: str) -> pd.DataFrame:
    """
    Loads sentiment data. 
    - Ignores 'ticker' column as requested.
    - Aggregates multiple entries per day into a single DAILY MEAN.
    - Returns columns: ['Date', 'Sentiment']
    """
    df = pd.read_csv(csv_path)
    
    # Ensure date is datetime
    # We assume the CSV has a 'date' column based on your description
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    # Group by Date and take the mean of the sentiment score
    # aggregating multiple entries for the same day.
    daily_df = df.groupby('Date')[['pos', 'neu', 'neg']].mean().reset_index()
    
    # Rename to match model feature naming convention
    daily_df = daily_df.rename(columns={'compound': 'Sentiment'})
    daily_df = daily_df.sort_values('Date')
    
    return daily_df

def fetch_option_chain(ticker: str, expiration: str):
    """Fetch option chain from yfinance for a single expiration."""
    t = yf.Ticker(ticker)
    oc = t.option_chain(expiration)
    calls = oc.calls.copy()
    calls['underlying_symbol'] = ticker
    calls['expiration'] = expiration
    return calls

def fetch_underlying_price(ticker: str):
    t = yf.Ticker(ticker)
    hist = t.history(period="5d")  # latest close
    if hist.empty:
        return None
    return float(hist['Close'].iloc[-1])

def compute_cond_volatility(ticker: str, window_days: int = 252, use_garch: bool = False):
    t = yf.Ticker(ticker)
    hist = t.history(period=f"{window_days}d")
    if hist.empty:
        return None
    returns = np.log(hist['Close']).diff().dropna()
    if use_garch and _HAS_ARCH:
        am = arch_model(returns * 100.0, vol='GARCH', p=1, q=1, dist='normal')
        res = am.fit(disp='off')
        cond_vol = res.conditional_volatility.iloc[-1] / 100.0
        return cond_vol * np.sqrt(252)
    else:
        vol = returns.std() * np.sqrt(252)
        return float(vol)

def yfinance_to_model_df(ticker: str, expiration: str, risk_free_rate: float = 0.0,
                         use_garch: bool = False, cond_vol_window: int = 252,
                         epu_df: pd.DataFrame = None, 
                         sentiment_df: pd.DataFrame = None):
    """
    Returns cleaned DataFrame with features used by your model.
    - epu_df: DataFrame containing 'Date' and 'EPU'
    - sentiment_df: DataFrame containing 'Date' and 'Sentiment'
    """
    calls = fetch_option_chain(ticker, expiration)
    if calls.empty:
        return pd.DataFrame()

    S = fetch_underlying_price(ticker)
    if S is None:
        S = np.nan

    df = calls.rename(columns={
        'impliedVolatility': 'IV',
        'lastPrice': 'Last',
        'strike': 'Strike',
        'volume': 'Volume',
        'openInterest': 'OpenInterest'
    }).copy()

    # Normalize IV
    df['IV'] = pd.to_numeric(df.get('IV', None), errors='coerce')
    df.loc[df['IV'] > 1, 'IV'] = df.loc[df['IV'] > 1, 'IV'] / 100.0

    df['underlying_stockprice'] = S
    df['option_price'] = pd.to_numeric(df.get('Last', None), errors='coerce')
    
    # Maturity
    try:
        today = pd.Timestamp.now(tz='UTC').date()
        delta = pd.to_datetime(expiration) - pd.to_datetime(today)
        maturity_days = int(delta.days)
    except Exception:
        maturity_days = 0

    maturity_days = max(maturity_days, 0)
    df['Maturity_days'] = maturity_days
    df['Maturity'] = df['Maturity_days'].apply(lambda d: _to_years(int(d)))

    df['K_over_S'] = df['Strike'] / df['underlying_stockprice']
    df['C_over_S'] = df['option_price'] / df['underlying_stockprice']
    df['Monyness'] = df['underlying_stockprice'] / df['Strike']
    df['r'] = risk_free_rate

    # Volatility
    if use_garch:
        cond_vol = compute_cond_volatility(ticker, window_days=cond_vol_window, use_garch=True)
        df['cond_vol'] = cond_vol
    else:
        cond_vol = compute_cond_volatility(ticker, window_days=cond_vol_window, use_garch=False)
        df['cond_vol'] = cond_vol

    # --- DATA MERGING ---
    # Check if we need to create a merge key (normalized date from lastTradeDate)
    needs_merge = (epu_df is not None and not epu_df.empty) or \
                  (sentiment_df is not None and not sentiment_df.empty)

    if needs_merge and 'lastTradeDate' in df.columns:
        # Normalize trade date to midnight UTC naive for matching
        df['temp_date'] = pd.to_datetime(df['lastTradeDate']).dt.tz_convert(None).dt.normalize()
        df = df.sort_values('temp_date')

        # 1. Merge EPU
        if epu_df is not None and not epu_df.empty:
            df = pd.merge_asof(df, epu_df, left_on='temp_date', right_on='Date', direction='backward')
            # Drop the right-side Date column from merge to avoid conflicts
            df = df.drop(columns=['Date'], errors='ignore')

        # 2. Merge Sentiment
        if sentiment_df is not None and not sentiment_df.empty:
            df = pd.merge_asof(df, sentiment_df, left_on='temp_date', right_on='Date', direction='backward')
            df = df.drop(columns=['Date'], errors='ignore')
        
        # Cleanup
        df = df.drop(columns=['temp_date'], errors='ignore')
    
    else:
        # Create empty columns if data missing, so feature selection doesn't crash
        if 'EPU' not in df.columns: df['EPU'] = np.nan
        if 'pos' not in df.columns: df['pos'] = np.nan
        if 'neu' not in df.columns: df['neu'] = np.nan
        if 'neg' not in df.columns: df['neg'] = np.nan
        # if 'Sentiment' not in df.columns: df['Sentiment'] = np.nan

    # Keep columns
    keep_cols = [
        'underlying_symbol','expiration','contractSymbol','Strike','underlying_stockprice',
        'option_price','Volume','OpenInterest','IV','Maturity','Maturity_days',
        'K_over_S','C_over_S','Monyness','r','cond_vol', 'EPU', 'pos', 'neu', 'neg'
    ]
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing]