import asyncio
import logging
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import alpaca_trade_api as tradeapi
from shared_state import lock, prices, positions, signals, equity_curve
from core import Portfolio


# ======================================================
# CONFIG
# ======================================================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

TICKERS = ["AAPL", "MSFT", "SPY"]
INITIAL_CAPITAL = 100_000
RISK_PER_TRADE = 0.01
MAX_DRAWDOWN = 0.10
STOP_LOSS_PCT = 0.03

FEATURE_COLS = ["atr", "ema20", "ema50", "rsi"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ======================================================
# CONNECT TO ALPACA
# ======================================================
# -------------------------------
# Alpaca Connection
# -------------------------------
alpaca = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")
account = alpaca.get_account()

logging.info(f"Connected to Alpaca | Cash: ${float(account.cash):,.2f}")

# -------------------------------
# Portfolio Initialization (REAL STATE)
# -------------------------------
portfolio = Portfolio(float(account.cash))

# -------------------------------
# Sync Existing Alpaca Positions
# -------------------------------
alpaca_positions = alpaca.list_positions()

for p in alpaca_positions:
    symbol = p.symbol
    qty = int(float(p.qty))
    entry_price = float(p.avg_entry_price)

    portfolio.positions[symbol] = {
        "qty": qty,
        "entry": entry_price
    }

    logging.info(
        f"🔄 Synced position: {symbol} | Qty: {qty} | Entry: {entry_price:.2f}"
    )


# ======================================================
# DATA + FEATURES
# ======================================================
def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def fetch_data(symbols, days=365):
    start = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    data = {}
    for s in symbols:
        bars = alpaca.get_bars(s, tradeapi.TimeFrame.Day, start=start).df
        bars.reset_index(inplace=True)
        bars["atr"] = bars["high"] - bars["low"]
        bars["ema20"] = bars["close"].ewm(span=20).mean()
        bars["ema50"] = bars["close"].ewm(span=50).mean()
        bars["rsi"] = rsi(bars["close"])
        bars.dropna(inplace=True)
        data[s] = bars
    return data

# ======================================================
# ML MODEL
# ======================================================
def train_model(data):
    df = pd.concat(data.values())
    df["regime"] = (df["atr"] > df["atr"].rolling(20).mean()).astype(int)
    df.dropna(inplace=True)

    X = df[FEATURE_COLS]
    y = df["regime"]

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    tscv = TimeSeriesSplit(5)

    scores = []
    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        scores.append(f1_score(y.iloc[te], model.predict(X.iloc[te])))

    logging.info(f"ML F1: {np.mean(scores):.3f}")
    model.fit(X, y)
    return model

# ======================================================
# PORTFOLIO + RISK
# ======================================================
class Portfolio:
    def __init__(self, cash):
        self.cash = cash
        self.positions = {}
        self.equity_peak = cash

    def equity(self, prices):
        equity = self.cash
        for s, p in self.positions.items():
            equity += p["qty"] * prices.get(s, p["entry"])
        return equity

    def open(self, symbol, qty, price):
        self.positions[symbol] = {"qty": qty, "entry": price}
        self.cash -= qty * price

    def close(self, symbol, price):
        pos = self.positions.pop(symbol)
        pnl = pos["qty"] * (price - pos["entry"])
        self.cash += pos["qty"] * price
        return pnl

    def drawdown(self, equity):
        self.equity_peak = max(self.equity_peak, equity)
        return (self.equity_peak - equity) / self.equity_peak

account = alpaca.get_account()
portfolio = Portfolio(float(account.cash))


# ======================================================
# EXECUTION
# ======================================================
def place_order(symbol, qty, side):
    if qty <= 0:
        return False
    alpaca.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type="market",
        time_in_force="day"
    )
    return True

# ======================================================
# STREAM HANDLER
# ======================================================
price_cache = {}

async def on_bar(bar):
    symbol = bar.symbol
    price = bar.close
    price_cache[symbol] = price

    df = data[symbol]
    last = df.iloc[-1].copy()
    extended_close = pd.concat(
    [df["close"], pd.Series([price])],
    ignore_index=True
)

    last["ema20"] = extended_close.ewm(span=20).mean().iloc[-1]
    last["ema50"] = extended_close.ewm(span=50).mean().iloc[-1]
    last["rsi"] = rsi(extended_close).iloc[-1]

    last["atr"] = bar.high - bar.low

    features = last[FEATURE_COLS].values.reshape(1, -1)
    regime = model.predict(features)[0]

    entry = last["close"] > last["ema20"] > last["ema50"]
    exit_ = last["close"] < last["ema20"]

    equity = portfolio.equity(price_cache)
    dd = portfolio.drawdown(equity)

    if dd > MAX_DRAWDOWN:
        logging.warning("MAX DRAWDOWN HIT — LIQUIDATING")
        for s in list(portfolio.positions.keys()):
            place_order(s, portfolio.positions[s]["qty"], "sell")
            portfolio.close(s, price_cache[s])
        return

    if symbol not in portfolio.positions and entry and regime == 1:
        risk_cash = portfolio.cash * RISK_PER_TRADE
        qty = int(risk_cash / price)
        if qty > 0:
            place_order(symbol, qty, "buy")
            portfolio.open(symbol, qty, price)
            logging.info(f"BUY {symbol} x{qty}")

    if symbol in portfolio.positions and exit_:
        qty = portfolio.positions[symbol]["qty"]
        place_order(symbol, qty, "sell")
        pnl = portfolio.close(symbol, price)
        logging.info(f"SELL {symbol} pnl={pnl:.2f}")

    if symbol in portfolio.positions:
        allow_entry = False
    else:
        allow_entry = True


    
    with lock:
        prices[symbol] = price
        equity_curve.append(portfolio.equity(prices))
        positions.clear()
        positions.update(portfolio.positions)

    if allow_entry and entry and regime == 1:
        signals.append({"symbol": symbol, "price": price, "type": "BUY", "time": datetime.now()})

    if exit_:
        signals.append({"symbol": symbol, "price": price, "type": "SELL", "time": datetime.now()})


# ======================================================
# RUN
# ======================================================
data = fetch_data(TICKERS)
model = train_model(data)

stream = tradeapi.Stream(API_KEY, API_SECRET, base_url=BASE_URL, data_feed="iex")
for t in TICKERS:
    stream.subscribe_bars(on_bar, t)

if abs(float(account.cash)) > float(account.equity):
    logging.warning("⚠️ High leverage detected. Risk controls tightened.")


logging.info("🚀 Phase 3 Engine Running")
try:
    stream.run()
except KeyboardInterrupt:
    logging.info("🛑 Keyboard interrupt received. Shutting down stream...")
    stream.stop()
    logging.info("🛑 Stream stopped. Exiting.")
