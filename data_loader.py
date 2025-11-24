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

def fetch_option_chain(ticker: str, expiration: str):
    """
    Fetch option chain from yfinance for a single expiration.
    expiration: string date 'YYYY-MM-DD' as returned by Ticker.options
    Returns: DataFrame similar to your CSVs (calls only). Columns include:
      - contractSymbol, lastPrice, strike, lastTradeDate, bid, ask, change,
        percentChange, volume, openInterest, impliedVolatility
    """
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
        # last conditional vol (in percent), convert to decimal annualized approx
        cond_vol = res.conditional_volatility.iloc[-1] / 100.0
        # approximate annualization:
        return cond_vol * np.sqrt(252)
    else:
        # use rolling std of returns (daily), then annualize
        vol = returns.std() * np.sqrt(252)
        return float(vol)

def yfinance_to_model_df(ticker: str, expiration: str, risk_free_rate: float = 0.0,
                         use_garch: bool = False, cond_vol_window: int = 252):
    """
    Returns cleaned DataFrame with features used by your model.
    - risk_free_rate: annual decimal (e.g., 0.017)
    """
    calls = fetch_option_chain(ticker, expiration)
    if calls.empty:
        return pd.DataFrame()

    S = fetch_underlying_price(ticker)
    if S is None:
        S = np.nan

    # map yfinance names -> notebook names
    df = calls.rename(columns={
        'impliedVolatility': 'IV',
        'lastPrice': 'Last',
        'strike': 'Strike',
        'volume': 'Volume',
        'openInterest': 'OpenInterest'
    }).copy()

    # ensure IV is decimal (yfinance often returns decimal already, but some sources give percent)
    # If values > 1, assume percent-like (e.g., 25) and divide by 100
    df['IV'] = pd.to_numeric(df.get('IV', None), errors='coerce')
    df.loc[df['IV'] > 1, 'IV'] = df.loc[df['IV'] > 1, 'IV'] / 100.0

    # Underlying price column used in your notebook
    df['underlying_stockprice'] = S
    df['option_price'] = pd.to_numeric(df.get('Last', None), errors='coerce')
    # compute scalar days until expiration (expiration is a single date string)
    # subtracting two Timestamps yields a Timedelta scalar, so use .days on that scalar
    try:
        # use a timezone-aware 'today' to avoid deprecation warnings and ensure correct subtraction
        today = pd.Timestamp.now(tz='UTC').date()
        delta = pd.to_datetime(expiration) - pd.to_datetime(today)
        maturity_days = int(delta.days)
    except Exception:
        # fallback: set to zero if parsing fails
        maturity_days = 0

    # clamp to non-negative days and assign as a constant column
    maturity_days = max(maturity_days, 0)
    df['Maturity_days'] = maturity_days
    df['Maturity'] = df['Maturity_days'].apply(lambda d: _to_years(int(d)))

    df['K_over_S'] = df['Strike'] / df['underlying_stockprice']
    df['C_over_S'] = df['option_price'] / df['underlying_stockprice']
    df['Monyness'] = df['underlying_stockprice'] / df['Strike']
    df['r'] = risk_free_rate

    # optional conditional volatility
    if use_garch:
        cond_vol = compute_cond_volatility(ticker, window_days=cond_vol_window, use_garch=True)
        df['cond_vol'] = cond_vol
    else:
        cond_vol = compute_cond_volatility(ticker, window_days=cond_vol_window, use_garch=False)
        df['cond_vol'] = cond_vol

    # keep columns your model expects (example list; adapt to notebook variable names)
    keep_cols = [
        'underlying_symbol','expiration','contractSymbol','Strike','underlying_stockprice',
        'option_price','Volume','OpenInterest','IV','Maturity','Maturity_days',
        'K_over_S','C_over_S','Monyness','r','cond_vol'
    ]
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing]
