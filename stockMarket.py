import logging
import os
import sys
import time
import threading
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score

from signals import ml_predict, rule_gate, BUY, SELL, HOLD
from state_store import (
    init_db, upsert_universe, upsert_signal, insert_decision,
    insert_risk_snapshot, upsert_model_health, upsert_engine_state,
    insert_order, update_order, insert_fill, upsert_position_snapshot
)
from alerts import alert_signal, alert_risk, alert_system

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- CONFIG ----------------
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
DATA_FEED = os.getenv("APCA_DATA_FEED", "iex")

WATCHLIST_FILE = os.getenv("WATCHLIST_FILE", "watchlist.txt")
UNIVERSE_MAX = int(os.getenv("UNIVERSE_MAX", "40"))
MIN_DOLLAR_VOL = float(os.getenv("MIN_DOLLAR_VOL", "20000000"))

UNIVERSE_REFRESH_SECONDS = int(os.getenv("UNIVERSE_REFRESH_SECONDS", "900"))  # 15 min
WATCHLIST_POLL_SECONDS = int(os.getenv("WATCHLIST_POLL_SECONDS", "5"))

# 5.10 safety
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "0").strip() == "1"  # default OFF
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.10"))
MAX_GROSS_EXPOSURE_PCT = float(os.getenv("MAX_GROSS_EXPOSURE_PCT", "1.25"))
EXPOSURE_REDUCE_STEP_PCT = float(os.getenv("EXPOSURE_REDUCE_STEP_PCT", "0.20"))
ENFORCE_COOLDOWN_SECONDS = int(os.getenv("ENFORCE_COOLDOWN_SECONDS", "30"))
RISK_SNAPSHOT_THROTTLE_SECONDS = int(os.getenv("RISK_SNAPSHOT_THROTTLE_SECONDS", "5"))

# 5.13 cooldown / spam protection
SYMBOL_COOLDOWN_SECONDS = int(os.getenv("SYMBOL_COOLDOWN_SECONDS", "90"))

# 5.14 sizing controls
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
MAX_POS_PCT_EQUITY = float(os.getenv("MAX_POS_PCT_EQUITY", "0.20"))  # max 20% equity per position
CORRELATION_THRESHOLD = float(os.getenv("CORRELATION_THRESHOLD", "0.75"))  # shrink when corr high
CORR_LOOKBACK_DAYS = int(os.getenv("CORR_LOOKBACK_DAYS", "60"))

# 5.15 stops/trailing
ATR_STOP_MULT = float(os.getenv("ATR_STOP_MULT", "2.0"))         # stop distance = ATR*mult
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "0.08"))                # 8% trailing
ENABLE_TRAIL = os.getenv("ENABLE_TRAIL", "1").strip() == "1"

MODEL_PATH = "models/ml_model.joblib"

if not API_KEY or not API_SECRET:
    logging.error("❌ Missing Alpaca API keys (APCA_API_KEY_ID / APCA_API_SECRET_KEY)")
    sys.exit(1)

alpaca = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# ---------------- DB INIT ----------------
init_db()

# ---------------- GLOBAL ENGINE STATE ----------------
SYMBOLS: list[str] = []
history: dict[str, pd.DataFrame] = {}
ml_model = None
cv_f1 = 0.0
last_train_ts = ""
initial_equity = None

portfolio = {
    "cash": 0.0,
    "positions": {},  # symbol -> {qty, entry_price, stop_price, trail_high}
}

# in-flight + cooldown controls
inflight = {}    # symbol -> {"side":..., "ts":...}
cooldowns = {}   # symbol -> last_trade_ts

# throttles / safety
_last_risk_snapshot_ts = 0.0
_last_enforce_ts = 0.0
_kill_switch_tripped = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------- UNIVERSE ----------------
def read_watchlist(path: str):
    symbols = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().upper()
                if s and not s.startswith("#"):
                    symbols.append(s)
    except FileNotFoundError:
        logging.warning(f"Watchlist not found: {path} (ok, using positions only)")
    return sorted(set(symbols))


def watchlist_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return 0.0


def universe_changed(old_syms, new_syms) -> bool:
    return set(old_syms) != set(new_syms)


def safe_get_snapshots(symbols):
    try:
        return alpaca.get_snapshots(symbols)
    except Exception as e:
        logging.warning(f"Snapshot fetch unavailable (liquidity filter disabled): {e}")
        return {}


def dollar_volume_from_snapshot(snap):
    try:
        v = float(snap.dailyBar.v)
        c = float(snap.dailyBar.c)
        return v * c
    except Exception:
        return 0.0


def filter_by_liquidity(symbols, min_dollar_vol):
    snaps = safe_get_snapshots(symbols)
    if not snaps:
        return symbols
    scored = []
    for s in symbols:
        snap = snaps.get(s)
        if not snap:
            continue
        dv = dollar_volume_from_snapshot(snap)
        if dv >= min_dollar_vol:
            scored.append((s, dv))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored]


def fetch_position_symbols():
    try:
        out = []
        for p in alpaca.list_positions():
            qty = int(float(p.qty))
            if qty != 0:
                out.append(p.symbol)
        return out
    except Exception:
        return []


def build_universe():
    held = set(fetch_position_symbols())
    watch = set(read_watchlist(WATCHLIST_FILE))

    filtered_watch = filter_by_liquidity(sorted(watch), MIN_DOLLAR_VOL)
    universe = sorted(held | set(filtered_watch))

    if len(universe) > UNIVERSE_MAX:
        remaining = max(0, UNIVERSE_MAX - len(held))
        if remaining == 0:
            universe = sorted(held)[:UNIVERSE_MAX]
        else:
            universe = sorted(held) + [s for s in filtered_watch if s not in held][:remaining]

    return sorted(dict.fromkeys(universe))


# ---------------- PORTFOLIO RESYNC + POS SNAPSHOT ----------------
def resync_from_broker():
    global portfolio, initial_equity

    try:
        acct = alpaca.get_account()
        cash = float(acct.cash)
        equity = float(acct.equity)

        if initial_equity is None:
            initial_equity = equity

        new_positions = {}
        try:
            for p in alpaca.list_positions():
                qty = int(float(p.qty))
                if qty == 0:
                    continue
                sym = p.symbol
                entry = float(p.avg_entry_price)

                # preserve existing stop/trail if we had it
                prev = portfolio["positions"].get(sym, {})
                stop_price = float(prev.get("stop_price", 0.0))
                trail_high = float(prev.get("trail_high", 0.0))

                new_positions[sym] = {
                    "qty": qty,
                    "entry_price": entry,
                    "stop_price": stop_price,
                    "trail_high": trail_high
                }

                upsert_position_snapshot(
                    symbol=sym,
                    qty=float(qty),
                    avg_entry_price=entry,
                    current_price=float(p.current_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(p.unrealized_pl),
                    stop_price=stop_price,
                    trail_high=trail_high
                )

        except Exception as e:
            logging.warning(f"Position list failed during resync: {e}")

        portfolio["cash"] = cash
        portfolio["positions"] = new_positions

        logging.info(f"✅ Resynced from broker | Cash: ${cash:,.2f} | Positions: {len(new_positions)}")
        for sym, pos in new_positions.items():
            logging.info(f"   🔄 {sym} | Qty: {pos['qty']} | Entry: {pos['entry_price']:.2f}")

        upsert_engine_state(
            trading_enabled=TRADING_ENABLED,
            kill_switch=_kill_switch_tripped,
            last_resync_ts=utc_now()
        )

    except Exception as e:
        logging.error(f"❌ Broker resync failed: {e}")
        alert_system("Broker resync failed", str(e))


# ---------------- INDICATORS ----------------
def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


FEATURE_COLS = ["atr", "ema20", "ema50", "rsi"]


# ---------------- HISTORICAL ----------------
def fetch_history(symbol, days=365):
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    bars = alpaca.get_bars(symbol, tradeapi.TimeFrame.Day, start=start).df
    if bars.empty:
        return None
    bars.reset_index(inplace=True)
    bars["atr"] = bars["high"] - bars["low"]
    bars["ema20"] = bars["close"].ewm(span=20).mean()
    bars["ema50"] = bars["close"].ewm(span=50).mean()
    bars["rsi"] = rsi(bars["close"])
    return bars.dropna()


def rebuild_history(symbols):
    new_hist = {}
    for s in symbols:
        try:
            df = fetch_history(s)
            if df is not None and len(df) > 60:
                new_hist[s] = df
        except Exception as e:
            logging.warning(f"History fetch failed for {s}: {e}")
    return new_hist


# ---------------- ML ----------------
def train_ml(history_dict):
    combined = pd.concat(history_dict.values(), ignore_index=True)
    combined["regime"] = (combined["atr"] > combined["atr"].rolling(20).mean()).astype(int)
    combined.dropna(inplace=True)

    X = combined[FEATURE_COLS]
    y = combined["regime"]

    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)

    scores = []
    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[te])
        scores.append(f1_score(y.iloc[te], preds))

    cv = float(np.mean(scores))
    logging.info(f"ML CV F1: {cv:.3f}")

    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model, cv


def rebuild_model(new_history):
    global ml_model, cv_f1, last_train_ts
    ml_model, cv_f1 = train_ml(new_history)
    last_train_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    upsert_model_health(cv_f1=cv_f1, last_train_ts=last_train_ts,
                        buy_rate=0.0, hold_rate=0.0, sell_rate=0.0, avg_confidence=0.0)


# ---------------- 5.14 Correlation scaling ----------------
def correlation_scale(symbol: str) -> float:
    """
    If symbol is highly correlated with existing held positions, scale down sizing.
    Uses daily history close returns.
    """
    held_syms = [s for s, p in portfolio["positions"].items() if int(p.get("qty", 0)) != 0]
    if not held_syms:
        return 1.0
    if symbol not in history:
        return 1.0

    # Build returns matrix
    cols = []
    df_map = {}

    for s in set(held_syms + [symbol]):
        if s in history:
            df_map[s] = history[s].copy()
            cols.append(s)

    if symbol not in df_map or len(cols) < 2:
        return 1.0

    prices = pd.DataFrame({s: df_map[s]["close"].tail(CORR_LOOKBACK_DAYS).reset_index(drop=True) for s in cols})
    rets = prices.pct_change().dropna()
    if rets.empty:
        return 1.0

    corr = rets.corr()
    scale = 1.0
    for h in held_syms:
        if h in corr.columns and symbol in corr.index:
            c = float(corr.loc[symbol, h])
            if c >= CORRELATION_THRESHOLD:
                # shrink by (1-c) but never below 20% floor
                scale *= max(0.2, (1.0 - c))
    return max(0.1, min(1.0, scale))


# ---------------- 5.15 Stop + trailing ----------------
def ensure_stops(symbol: str, atr: float, price: float):
    """
    Initialize or update trailing info for held positions.
    """
    if symbol not in portfolio["positions"]:
        return
    pos = portfolio["positions"][symbol]
    qty = int(pos.get("qty", 0))
    if qty <= 0:
        return

    entry = float(pos["entry_price"])
    stop_price = float(pos.get("stop_price", 0.0))
    trail_high = float(pos.get("trail_high", 0.0))

    # init trail high
    trail_high = max(trail_high, price)

    # init stop if missing
    if stop_price <= 0.0:
        stop_price = max(0.01, entry - ATR_STOP_MULT * atr)

    # trailing logic
    if ENABLE_TRAIL:
        trail_high = max(trail_high, price)
        trail_stop = trail_high * (1.0 - TRAIL_PCT)
        stop_price = max(stop_price, trail_stop)

    pos["stop_price"] = float(stop_price)
    pos["trail_high"] = float(trail_high)

    # snapshot write (dashboard)
    upsert_position_snapshot(
        symbol=symbol,
        qty=float(qty),
        avg_entry_price=float(entry),
        current_price=float(price),
        market_value=float(qty * price),
        unrealized_pl=float((price - entry) * qty),
        stop_price=float(stop_price),
        trail_high=float(trail_high),
    )


def stop_triggered(symbol: str, price: float) -> bool:
    pos = portfolio["positions"].get(symbol)
    if not pos:
        return False
    stop_price = float(pos.get("stop_price", 0.0))
    return stop_price > 0.0 and price <= stop_price


# ---------------- Risk snapshots + enforcement ----------------
def publish_risk_snapshot(throttle=True):
    global initial_equity, _last_risk_snapshot_ts
    now = time.time()
    if throttle and (now - _last_risk_snapshot_ts) < RISK_SNAPSHOT_THROTTLE_SECONDS:
        return None
    _last_risk_snapshot_ts = now

    acct = alpaca.get_account()
    equity = float(acct.equity)
    cash = float(acct.cash)
    buying_power = float(acct.buying_power)

    if initial_equity is None:
        initial_equity = equity

    unrealized = 0.0
    gross = 0.0
    net = 0.0

    try:
        positions = alpaca.list_positions()
        for p in positions:
            mv = float(p.market_value)
            upl = float(p.unrealized_pl)
            unrealized += upl
            gross += abs(mv)
            net += mv
    except Exception as e:
        alert_system("Risk snapshot positions read failed", str(e))

    invested_pct = (gross / equity) if equity > 0 else 0.0
    drawdown_pct = max(0.0, (initial_equity - equity) / initial_equity) if initial_equity and initial_equity > 0 else 0.0

    insert_risk_snapshot(
        equity=equity, cash=cash, buying_power=buying_power,
        gross_exposure=gross, net_exposure=net, invested_pct=invested_pct,
        unrealized_pl=unrealized, realized_pl=0.0, drawdown_pct=drawdown_pct
    )

    if drawdown_pct >= MAX_DRAWDOWN:
        alert_risk("MAX DRAWDOWN WARNING", f"drawdown={drawdown_pct:.2%} >= {MAX_DRAWDOWN:.2%}", level="ERROR")

    if invested_pct > MAX_GROSS_EXPOSURE_PCT:
        alert_risk("EXPOSURE CAP WARNING", f"gross/equity={invested_pct:.2f} > {MAX_GROSS_EXPOSURE_PCT:.2f}", level="WARN")

    return {"equity": equity, "cash": cash, "invested_pct": invested_pct, "drawdown_pct": drawdown_pct}


def liquidate_all_positions(reason: str):
    global _kill_switch_tripped
    if _kill_switch_tripped:
        return
    _kill_switch_tripped = True
    alert_risk("FORCED LIQUIDATION", reason, level="ERROR")
    logging.error(f"🧨 FORCED LIQUIDATION: {reason}")

    upsert_engine_state(trading_enabled=TRADING_ENABLED, kill_switch=True, last_enforce_ts=utc_now(), last_enforce_action="LIQUIDATE_ALL")

    try:
        for p in alpaca.list_positions():
            qty = int(float(p.qty))
            if qty == 0:
                continue
            side = "sell" if qty > 0 else "buy"
            place_order(p.symbol, abs(qty), side, submitted_price=float(p.current_price), reason=reason)
    except Exception as e:
        alert_system("Liquidation failed", str(e))

    resync_from_broker()


def reduce_exposure_step(reason: str):
    """
    Reduce the largest position (by market value) by EXPOSURE_REDUCE_STEP_PCT.
    """
    global _last_enforce_ts

    now = time.time()
    if (now - _last_enforce_ts) < ENFORCE_COOLDOWN_SECONDS:
        return
    _last_enforce_ts = now

    try:
        positions = alpaca.list_positions()
        rows = []
        for p in positions:
            qty = int(float(p.qty))
            if qty == 0:
                continue
            rows.append((p.symbol, abs(float(p.market_value)), qty, float(p.current_price)))

        if not rows:
            return

        rows.sort(key=lambda x: x[1], reverse=True)
        sym, mv_abs, qty, px = rows[0]
        reduce_qty = max(1, int(abs(qty) * EXPOSURE_REDUCE_STEP_PCT))
        side = "sell" if qty > 0 else "buy"

        upsert_engine_state(trading_enabled=TRADING_ENABLED, kill_switch=_kill_switch_tripped,
                            last_enforce_ts=utc_now(), last_enforce_action=f"REDUCE_EXPOSURE {sym} {reduce_qty}")

        place_order(sym, reduce_qty, side, submitted_price=px, reason=reason)

    except Exception as e:
        alert_system("Exposure reduction failed", str(e))


def enforce_risk(metrics: dict | None):
    if not metrics:
        return

    dd = float(metrics["drawdown_pct"])
    invested = float(metrics["invested_pct"])

    if dd >= MAX_DRAWDOWN:
        liquidate_all_positions(f"Drawdown {dd:.2%} >= {MAX_DRAWDOWN:.2%}")
        return

    if invested > MAX_GROSS_EXPOSURE_PCT:
        reduce_exposure_step(f"gross/equity {invested:.2f} > cap {MAX_GROSS_EXPOSURE_PCT:.2f}")


# ---------------- 5.12 Orders + reconciliation ----------------
def can_trade_symbol(symbol: str) -> tuple[bool, str]:
    # in-flight
    if symbol in inflight:
        return False, "In-flight order exists"

    # cooldown
    last = cooldowns.get(symbol, 0.0)
    if time.time() - last < SYMBOL_COOLDOWN_SECONDS:
        return False, f"Cooldown active ({SYMBOL_COOLDOWN_SECONDS}s)"

    return True, "OK"


def place_order(symbol: str, qty: int, side: str, submitted_price: float, reason: str) -> bool:
    """
    Creates a client_order_id, submits order (if enabled), stores to DB.
    In monitor-only mode, logs & stores as SIMULATED_SKIP.
    """
    qty = int(qty)
    if qty <= 0:
        return False

    ok, why = can_trade_symbol(symbol)
    if not ok:
        insert_decision(symbol, float(submitted_price), "SKIP", "ORDER_GUARD", 1.0, False, why)
        return False

    client_order_id = f"eng-{uuid.uuid4().hex[:16]}"
    ts = utc_now()

    if not TRADING_ENABLED:
        logging.info(f"🛑 TRADING_DISABLED: would {side.upper()} {symbol} x{qty}")
        insert_order(ts, order_id="SIMULATED", client_order_id=client_order_id, symbol=symbol, side=side,
                     qty=float(qty), otype="market", tif="day", status="SIMULATED_SKIP", submitted_price=float(submitted_price))
        cooldowns[symbol] = time.time()
        return False

    try:
        inflight[symbol] = {"side": side, "ts": time.time()}

        o = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
            client_order_id=client_order_id
        )

        order_id = getattr(o, "id", "") or ""
        status = getattr(o, "status", "submitted")

        insert_order(ts, order_id=order_id, client_order_id=client_order_id, symbol=symbol, side=side,
                     qty=float(qty), otype="market", tif="day", status=status, submitted_price=float(submitted_price))

        cooldowns[symbol] = time.time()
        return True

    except Exception as e:
        alert_system("Order failed", f"{symbol} {side} qty={qty} err={e}")
        insert_order(ts, order_id="", client_order_id=client_order_id, symbol=symbol, side=side,
                     qty=float(qty), otype="market", tif="day", status="ERROR",
                     submitted_price=float(submitted_price), error=str(e))
        return False


async def trade_updates_handler(data):
    try:
        event = getattr(data, "event", None) or data.get("event")
        order = getattr(data, "order", None) or data.get("order", {})

        order_id = order.get("id") if isinstance(order, dict) else getattr(order, "id", "")
        symbol = order.get("symbol") if isinstance(order, dict) else getattr(order, "symbol", "")
        side = order.get("side") if isinstance(order, dict) else getattr(order, "side", "")
        status = order.get("status") if isinstance(order, dict) else getattr(order, "status", "")

        filled_qty = float(order.get("filled_qty", 0.0)) if isinstance(order, dict) else float(getattr(order, "filled_qty", 0.0) or 0.0)
        filled_avg = float(order.get("filled_avg_price", 0.0)) if isinstance(order, dict) else float(getattr(order, "filled_avg_price", 0.0) or 0.0)

        if order_id:
            update_order(order_id, status=status or str(event), filled_qty=filled_qty, filled_avg_price=filled_avg)

        if event in ("fill", "partial_fill") and order_id and symbol:
            last_price = filled_avg if filled_avg > 0 else 0.0
            insert_fill(order_id, symbol, side, filled_qty, last_price)

        if symbol and status in ("filled", "canceled", "rejected", "expired", "done_for_day"):
            inflight.pop(symbol, None)

        if status == "filled":
            resync_from_broker()

    except Exception as e:
        logging.warning(f"trade_updates_handler error: {e}")


def rest_reconcile_open_orders_loop():
    """
    REST fallback: every few seconds, check any open in-flight symbols and clear if broker reports terminal.
    """
    while True:
        try:
            time.sleep(5)
            if not inflight:
                continue
            # If we have in-flight, pull open orders for those symbols
            syms = list(inflight.keys())
            try:
                open_orders = alpaca.list_orders(status="open", limit=200)
            except Exception:
                open_orders = []

            open_syms = set()
            for o in open_orders:
                open_syms.add(getattr(o, "symbol", ""))

            for s in syms:
                if s not in open_syms:
                    # likely filled/canceled; clear lock and resync
                    inflight.pop(s, None)
                    resync_from_broker()

        except Exception:
            pass


# ---------------- 5.14 sizing ----------------
def compute_qty(symbol: str, price: float, equity: float) -> int:
    """
    Risk sizing with:
    - RISK_PER_TRADE of cash (not margin)
    - MAX_POS_PCT_EQUITY cap
    - correlation scaling vs held positions
    - exposure headroom cap (MAX_GROSS_EXPOSURE_PCT)
    """
    cash_for_risk = max(0.0, portfolio["cash"])
    risk_budget = cash_for_risk * RISK_PER_TRADE

    # hard cap per position by equity
    pos_cap_value = equity * MAX_POS_PCT_EQUITY

    # correlation scaling
    scale = correlation_scale(symbol)
    risk_budget *= scale
    pos_cap_value *= scale

    # estimate headroom vs exposure cap
    # (approx) current gross from broker snapshot (fast) using account values in risk_snap
    # For simplicity, use equity cap and cash budget here; enforcement will handle remaining.
    target_value = min(risk_budget, pos_cap_value)

    qty = int(target_value / price) if price > 0 else 0
    return max(0, qty)


# ---------------- STREAM HANDLER ----------------
async def on_bar(bar):
    global history

    symbol = bar.symbol
    price = float(bar.close)

    if symbol not in history:
        return

    df = history[symbol]
    closes = pd.concat([df["close"], pd.Series([price])], ignore_index=True)

    atr = float(getattr(bar, "high", price) - getattr(bar, "low", price))
    ema20 = float(closes.ewm(span=20).mean().iloc[-1])
    ema50 = float(closes.ewm(span=50).mean().iloc[-1])
    rsi_val = float(rsi(closes).iloc[-1])

    # ML features as DataFrame
    features_df = pd.DataFrame([{"atr": atr, "ema20": ema20, "ema50": ema50, "rsi": rsi_val}])

    ml_signal, p_buy, p_sell = ml_predict(ml_model, features_df)
    rule_ok, reason = rule_gate(price, ema20, ema50, rsi_val, ml_signal, p_buy, p_sell)

    stored_conf = float(p_buy) if ml_signal == BUY else (float(p_sell) if ml_signal == SELL else float(max(p_buy, p_sell)))

    upsert_signal(
        symbol=symbol,
        price=price,
        ema20=ema20,
        ema50=ema50,
        ml_signal=ml_signal,
        confidence=stored_conf,
        rule_ok=bool(rule_ok),
        reason=reason
    )

    if ml_signal in (BUY, SELL) and (stored_conf >= 0.65):
        alert_signal(symbol, ml_signal, stored_conf, rule_ok, reason)

    # Update stop/trailing snapshots for held positions
    ensure_stops(symbol, atr=atr, price=price)

    # If kill-switch is tripped, do not trade
    if _kill_switch_tripped:
        insert_decision(symbol, price, "SKIP", ml_signal, stored_conf, bool(rule_ok), "Kill switch tripped")
        metrics = publish_risk_snapshot(throttle=True)
        upsert_engine_state(TRADING_ENABLED, True)
        enforce_risk(metrics)
        return

    # STOP LOSS / trailing exit (highest priority)
    if symbol in portfolio["positions"] and stop_triggered(symbol, price):
        qty = int(portfolio["positions"][symbol]["qty"])
        if qty > 0:
            ok = place_order(symbol, qty, "sell", submitted_price=price, reason="STOP/TRAIL hit")
            insert_decision(symbol, price, "SELL_STOP", "STOP", 1.0, True, "STOP/TRAIL hit")
            if ok:
                portfolio["positions"].pop(symbol, None)
        else:
            portfolio["positions"].pop(symbol, None)

    # ML EXIT (long-only SELL)
    if ml_signal == SELL and rule_ok and symbol in portfolio["positions"]:
        qty = int(portfolio["positions"][symbol]["qty"])
        if qty > 0:
            ok = place_order(symbol, qty, "sell", submitted_price=price, reason=reason)
            insert_decision(symbol, price, "SELL", ml_signal, float(p_sell), True, reason)
            if ok:
                portfolio["positions"].pop(symbol, None)

    # Entry (BUY only)
    if ml_signal == BUY and rule_ok and symbol not in portfolio["positions"]:
        metrics = publish_risk_snapshot(throttle=True) or {}
        equity = float(metrics.get("equity", 0.0)) if metrics else 0.0
        qty = compute_qty(symbol, price, equity) if equity > 0 else 0

        if qty <= 0:
            insert_decision(symbol, price, "SKIP", ml_signal, float(p_buy), False, "Qty=0 (sizing)")
        else:
            ok = place_order(symbol, qty, "buy", submitted_price=price, reason=reason)
            insert_decision(symbol, price, "BUY", ml_signal, float(p_buy), True, reason)
            if ok:
                # initialize stop/trail immediately using current atr
                portfolio["positions"][symbol] = {
                    "qty": qty,
                    "entry_price": price,
                    "stop_price": max(0.01, price - ATR_STOP_MULT * atr),
                    "trail_high": price
                }

    # Always publish risk + enforce
    metrics = publish_risk_snapshot(throttle=True)
    enforce_risk(metrics)

    upsert_engine_state(
        trading_enabled=TRADING_ENABLED,
        kill_switch=_kill_switch_tripped
    )


# ---------------- STREAM LIFECYCLE ----------------
def _run_stream_in_thread(stream_obj):
    try:
        stream_obj.run()
    except Exception as e:
        logging.error(f"Stream thread crashed: {e}")
        alert_system("Stream crashed", str(e))


def safe_stop_stream(stream_obj):
    for method in ("stop", "stop_ws", "close", "stop_stream"):
        fn = getattr(stream_obj, method, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                pass


def build_stream(symbols):
    s = Stream(API_KEY, API_SECRET, base_url=BASE_URL, data_feed=DATA_FEED)
    for sym in symbols:
        s.subscribe_bars(on_bar, sym)
    # Trade updates for order tracking (5.12)
    s.subscribe_trade_updates(trade_updates_handler)
    return s


def supervisor_loop():
    global SYMBOLS, history

    last_refresh = time.time()
    last_mtime = watchlist_mtime(WATCHLIST_FILE)

    SYMBOLS = build_universe()
    logging.info(f"🎯 ACTIVE UNIVERSE ({len(SYMBOLS)}): {SYMBOLS}")
    upsert_universe(SYMBOLS)

    upsert_engine_state(
        trading_enabled=TRADING_ENABLED,
        kill_switch=_kill_switch_tripped,
        last_universe_refresh_ts=utc_now()
    )

    # Resync broker truth at startup
    resync_from_broker()

    # Build history + train model
    history = rebuild_history(SYMBOLS)
    if not history:
        logging.error("❌ No historical data loaded. Check Alpaca data access/feed.")
        sys.exit(1)

    rebuild_model(history)
    logging.info(f"🧠 Model trained | CV F1={cv_f1:.3f} | last_train_ts={last_train_ts}")
    logging.info(f"🔒 TRADING_ENABLED={TRADING_ENABLED} (set env TRADING_ENABLED=1 to allow orders)")

    # REST reconcile background (5.12 fallback)
    threading.Thread(target=rest_reconcile_open_orders_loop, daemon=True).start()

    # Start stream thread
    stream_obj = build_stream(SYMBOLS)
    t = threading.Thread(target=_run_stream_in_thread, args=(stream_obj,), daemon=True)
    t.start()

    while True:
        time.sleep(WATCHLIST_POLL_SECONDS)

        current_mtime = watchlist_mtime(WATCHLIST_FILE)
        time_due = (time.time() - last_refresh) >= UNIVERSE_REFRESH_SECONDS

        if current_mtime != last_mtime or time_due:
            last_mtime = current_mtime
            last_refresh = time.time()

            new_symbols = build_universe()

            upsert_engine_state(
                trading_enabled=TRADING_ENABLED,
                kill_switch=_kill_switch_tripped,
                last_universe_refresh_ts=utc_now()
            )

            if universe_changed(SYMBOLS, new_symbols):
                logging.info("🔄 Universe change detected. Restarting stream safely...")
                logging.info(f"Old ({len(SYMBOLS)}): {SYMBOLS}")
                logging.info(f"New ({len(new_symbols)}): {new_symbols}")

                safe_stop_stream(stream_obj)
                t.join(timeout=10)

                SYMBOLS = new_symbols
                upsert_universe(SYMBOLS)

                resync_from_broker()

                new_history = rebuild_history(SYMBOLS)
                if new_history:
                    history = new_history
                    rebuild_model(history)
                    logging.info(f"🧠 Model retrained | CV F1={cv_f1:.3f} | last_train_ts={last_train_ts}")

                stream_obj = build_stream(SYMBOLS)
                t = threading.Thread(target=_run_stream_in_thread, args=(stream_obj,), daemon=True)
                t.start()

            else:
                logging.info("🔄 Universe refresh: no changes.")


if __name__ == "__main__":
    try:
        logging.info("🚀 Trading engine running — Phase 5.11–5.15 (Execution+Sizing+Stops)")
        supervisor_loop()
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
