import asyncio
import logging
import sys
import os
import creds
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import joblib
import alpaca_trade_api as tradeapi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------- CONFIG ----------------
# Load API credentials from environment if available (preferred)
API_KEY = os.environ.get('ALPACA_API_KEY') or os.environ.get('APCA_API_KEY_ID') or creds.API_KEY
API_SECRET = os.environ.get('ALPACA_SECRET_KEY') or os.environ.get('APCA_API_SECRET_KEY') or creds.SECRET_KEY
BASE_URL = os.environ.get('ALPACA_BASE_URL', "https://paper-api.alpaca.markets/")
TICKERS = ["AAPL","MSFT","SPY"]
INITIAL_CAPITAL = 100_000
RISK_PER_TRADE = 0.01
MAX_DRAWDOWN = 0.05
FEATURE_COLS = ['atr','ema20','ema50','rsi']

# ---------------- CONNECT TO ALPACA ----------------
try:
    alpaca = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    account = alpaca.get_account()
    logging.info("✅ Connected to Alpaca Paper Account")
    logging.info(f"Cash Available: ${float(account.cash):,.2f}")
except Exception as e:
    logging.error(f"❌ Failed to connect to Alpaca: {e}")
    sys.exit(1)

# If credentials are clearly placeholders or missing, give a helpful error before trying to connect
if not API_KEY or API_KEY.startswith("YOUR_") or not API_SECRET or API_SECRET.startswith("YOUR_"):
    logging.error(
        "Alpaca API credentials missing or placeholder detected.\n"
        "Set environment variables `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`,\n"
        "or replace the placeholders in stockMarket.py with your paper API keys.\n"
        "If you use VS Code, add the env vars to your launch configuration or system/user environment."
    )
    sys.exit(1)

# ---------------- PORTFOLIO STATE ----------------
portfolio = {
    "cash": INITIAL_CAPITAL,
    "shares": {t:0 for t in TICKERS},
    "equity_curve": [],
    "history": []
}

# ---------------- STOCK ORDER UTILITY ----------------
def place_stock_order(symbol, qty, side='buy'):
    try:
        if qty <= 0:
            return False
        logging.info(f"Attempting {side.upper()} {symbol} x {qty}")
        alpaca.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='day')
        logging.info(f"Order executed: {side.upper()} {symbol} x {qty}")
        return True
    except Exception as e:
        logging.error(f"Order failed for {symbol}: {e}")
        return False

# ---------------- FETCH HISTORICAL DATA ----------------
import concurrent.futures
import time


def rsi(series, window=14):
    """Compute the Relative Strength Index (RSI) for a pandas Series."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi_values = 100 - (100 / (1 + rs))
    return rsi_values.fillna(50)


def _get_bars_with_timeout(symbol, start_date, timeout=10):
    """Run alpaca.get_bars in a thread and enforce a timeout (avoids blocking the main thread).
    Raises TimeoutError on timeout, or re-raises underlying exceptions."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(alpaca.get_bars, symbol, tradeapi.TimeFrame.Day, start=start_date)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"Timeout fetching bars for {symbol} after {timeout}s")


def fetch_data(tickers, days=365, timeout=10, retries=3, backoff_base=1.0):
    df_dict = {}
    # use a date-only start (YYYY-MM-DD) to satisfy Alpaca's expected formats
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    for t in tickers:
        attempt = 0
        backoff = backoff_base
        bars = None
        while attempt < retries:
            try:
                bars = _get_bars_with_timeout(t, start_date, timeout=timeout)
                break
            except TimeoutError as e: 
                attempt += 1
                logging.warning(f"Timeout fetching {t} (attempt {attempt}/{retries}): {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
            except Exception as e:
                attempt += 1
                logging.warning(f"Error fetching {t} (attempt {attempt}/{retries}): {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
        if bars is None:
            logging.error(f"Failed to fetch bars for {t} after {retries} attempts. Skipping.")
            continue
        df = bars.df
        df.reset_index(inplace=True)
        df['atr'] = df['high']-df['low']
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['symbol'] = t
        df['rsi'] = rsi(df['close'])
        df['entry_signal'] = (df['close']>df['ema20']) & (df['ema20']>df['ema50'])
        df['exit_signal'] = df['close']<df['ema20']
        df_dict[t] = df
    return df_dict

df_dict = fetch_data(TICKERS)

# ---------------- ML MODEL ----------------
def train_ml(df_dict):
    # Concatenate while preserving symbol and timestamp information
    combined = pd.concat(df_dict.values(), ignore_index=True)
    # Normalize timestamp column name if present
    if 'index' in combined.columns and 'timestamp' not in combined.columns:
        combined.rename(columns={'index': 'timestamp'}, inplace=True)
    if 'timestamp' in combined.columns:
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        combined.sort_values('timestamp', inplace=True)

    combined['regime'] = (combined['atr'] > combined['atr'].rolling(20).mean()).astype(int)
    combined.dropna(inplace=True)

    X = combined[FEATURE_COLS]
    y = combined['regime']

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)

    scores = []
    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        scores.append(f1_score(y.iloc[test_idx], preds))

    logging.info(f"TimeSeries CV F1: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    # Retrain on full dataset and persist model
    model.fit(X, y)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/ml_model.joblib')
    logging.info('Saved trained ML model to models/ml_model.joblib')

    return model

ml_model = train_ml(df_dict)

# ---------------- LIVE BAR HANDLER ----------------
async def handle_bar(bar):
    t = bar.symbol
    price = bar.close
    df = df_dict[t]
    latest_row = df.iloc[-1]
    try:
        regime = ml_model.predict([latest_row[FEATURE_COLS].values])[0]
        # Entry
        if latest_row['entry_signal'] and regime==1 and portfolio['shares'][t]==0:
            qty = int((portfolio['cash']*RISK_PER_TRADE)/price)
            if qty>0 and place_stock_order(t, qty,'buy'):
                portfolio['shares'][t] += qty
                portfolio['cash'] -= qty*price
                portfolio['history'].append(f"BUY {qty} {t}@{price:.2f}")
        # Exit
        if latest_row['exit_signal'] and portfolio['shares'][t]>0:
            qty = portfolio['shares'][t]
            if place_stock_order(t, qty,'sell'):
                portfolio['shares'][t]=0
                portfolio['cash']+=qty*price
                portfolio['history'].append(f"SELL {qty} {t}@{price:.2f}")
    except Exception as e:
        logging.error(f"Error processing bar for {t}: {e}")

# ---------------- START STREAM ----------------
stream = tradeapi.Stream(API_KEY, API_SECRET, base_url=BASE_URL, data_feed='iex')
for t in TICKERS:
    stream.subscribe_bars(handle_bar, t)

# ---------------- RUN ----------------
if __name__=="__main__":
    try:
        logging.info("Starting trading engine (paper mode)...")
        # Use asyncio.run to create and manage the event loop (avoids "no current event loop" on some Python versions)
        asyncio.run(stream.run())
    except KeyboardInterrupt:
        logging.info("Shutdown requested (KeyboardInterrupt). Exiting...")
    except Exception as e:
        logging.error(f"Unexpected error running the stream: {e}")
        raise
