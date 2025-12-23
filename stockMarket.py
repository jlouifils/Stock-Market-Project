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
from core.risk import RiskManager
from core.portfolio import Portfolio
from core.alerts import send_alert


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

risk = RiskManager(
    max_drawdown=0.10,
    stop_loss_pct=0.03
)

portfolio = Portfolio(initial_cash=INITIAL_CAPITAL)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    symbol = bar.symbol
    price = bar.close

    # Get dataframe and base last row for feature construction
    df = df_dict[symbol]
    last = df.iloc[-1].copy()

    # Update last row with new bar values (if present on bar)
    last['close'] = price
    if hasattr(bar, 'high'):
        last['high'] = bar.high
    if hasattr(bar, 'low'):
        last['low'] = bar.low

    # Recompute derived features using recent history + this bar
    recent_closes = pd.concat([df['close'], pd.Series([price])], ignore_index=True)
    last['atr'] = last['high'] - last['low']
    last['ema20'] = recent_closes.ewm(span=20).mean().iloc[-1]
    last['ema50'] = recent_closes.ewm(span=50).mean().iloc[-1]
    last['rsi'] = rsi(recent_closes).iloc[-1]

    entry_signal = (last['close'] > last['ema20']) and (last['ema20'] > last['ema50'])
    exit_signal = last['close'] < last['ema20']

    try:
        features = last[FEATURE_COLS].values.reshape(1, -1)
        regime = ml_model.predict(features)[0]

        # ENTRY
        if entry_signal and regime == 1 and symbol not in portfolio.positions:
            qty = int((portfolio.cash * RISK_PER_TRADE) / price)
            if qty > 0:
                position_value = qty * price
                total_equity = portfolio.total_equity({s: price for s in TICKERS})
                if risk.position_size_allowed(position_value, total_equity) and \
                   risk.portfolio_exposure_allowed(portfolio.total_exposure({s: price for s in TICKERS}) + position_value, total_equity):
                    if place_stock_order(symbol, qty, 'buy'):
                        portfolio.open_position(symbol, qty, price)
                        logging.info(f"Opened {symbol} x{qty} @ {price:.2f}")

        # EXIT
        if exit_signal and symbol in portfolio.positions:
            pos = portfolio.positions[symbol]
            qty = pos['qty']
            if place_stock_order(symbol, qty, 'sell'):
                pnl = portfolio.close_position(symbol, price, "EXIT_SIGNAL")
                logging.info(f"Closed {symbol} x{qty} @ {price:.2f} pnl={pnl:.2f}")

    except Exception as e:
        logging.error(f"Error processing bar for {symbol}: {e}")

    # Update risk & enforce safety checks using current price_map
    price_map = {s: price for s in TICKERS}
    equity = portfolio.total_equity(price_map)
    risk.update_equity(equity)

    # HARD KILL SWITCH
    if risk.check_drawdown():
        for sym, pos in list(portfolio.positions.items()):
            place_stock_order(sym, pos["qty"], "sell")
            portfolio.close_position(sym, price, "MAX_DRAWDOWN")
            send_alert("FORCED LIQUIDATION — MAX DRAWDOWN", "RISK")
        return

    # STOP-LOSS check
    for sym, pos in list(portfolio.positions.items()):
        if risk.check_stop_loss(pos["entry_price"], price):
            place_stock_order(sym, pos["qty"], "sell")
            portfolio.close_position(sym, price, "STOP_LOSS")
            send_alert(f"STOP LOSS HIT — {sym}", "RISK")


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
