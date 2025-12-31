# state_store.py
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.getenv("STATE_DB", "state.db")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = _connect()
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS universe(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbols TEXT NOT NULL,
        updated_ts TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        level TEXT NOT NULL,
        category TEXT NOT NULL,
        title TEXT NOT NULL,
        message TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals(
        symbol TEXT PRIMARY KEY,
        price REAL,
        ema20 REAL,
        ema50 REAL,
        ml_signal TEXT,
        confidence REAL,
        rule_ok INTEGER,
        reason TEXT,
        updated_ts TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS decisions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        symbol TEXT NOT NULL,
        price REAL NOT NULL,
        action TEXT NOT NULL,
        ml_signal TEXT,
        confidence REAL,
        rule_ok INTEGER,
        reason TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS risk_snapshots(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        equity REAL,
        cash REAL,
        buying_power REAL,
        gross_exposure REAL,
        net_exposure REAL,
        invested_pct REAL,
        unrealized_pl REAL,
        realized_pl REAL,
        drawdown_pct REAL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS model_health(
        id INTEGER PRIMARY KEY CHECK (id=1),
        cv_f1 REAL,
        last_train_ts TEXT,
        buy_rate REAL,
        hold_rate REAL,
        sell_rate REAL,
        avg_confidence REAL,
        updated_ts TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS engine_state(
        id INTEGER PRIMARY KEY CHECK (id=1),
        trading_enabled INTEGER,
        kill_switch INTEGER,
        last_resync_ts TEXT,
        last_enforce_ts TEXT,
        last_enforce_action TEXT,
        last_universe_refresh_ts TEXT,
        updated_ts TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS orders(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        order_id TEXT,
        client_order_id TEXT,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        qty REAL NOT NULL,
        type TEXT,
        tif TEXT,
        status TEXT,
        submitted_price REAL,
        filled_qty REAL,
        filled_avg_price REAL,
        error TEXT,
        updated_ts TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS fills(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        order_id TEXT,
        symbol TEXT,
        side TEXT,
        qty REAL,
        price REAL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS positions_snapshot(
        symbol TEXT PRIMARY KEY,
        qty REAL,
        avg_entry_price REAL,
        current_price REAL,
        market_value REAL,
        unrealized_pl REAL,
        stop_price REAL,
        trail_high REAL,
        updated_ts TEXT
    );
    """)

    con.commit()
    con.close()


def upsert_universe(symbols: List[str]):
    con = _connect()
    con.execute("INSERT INTO universe(symbols, updated_ts) VALUES(?,?)", (",".join(symbols), _utc_now()))
    con.commit()
    con.close()


def upsert_signal(symbol: str, price: float, ema20: float, ema50: float,
                  ml_signal: str, confidence: float, rule_ok: bool, reason: str):
    con = _connect()
    con.execute("""
    INSERT INTO signals(symbol,price,ema20,ema50,ml_signal,confidence,rule_ok,reason,updated_ts)
    VALUES(?,?,?,?,?,?,?,?,?)
    ON CONFLICT(symbol) DO UPDATE SET
        price=excluded.price,
        ema20=excluded.ema20,
        ema50=excluded.ema50,
        ml_signal=excluded.ml_signal,
        confidence=excluded.confidence,
        rule_ok=excluded.rule_ok,
        reason=excluded.reason,
        updated_ts=excluded.updated_ts;
    """, (symbol, price, ema20, ema50, ml_signal, confidence, int(rule_ok), reason, _utc_now()))
    con.commit()
    con.close()


def insert_decision(symbol: str, price: float, action: str, ml_signal: str,
                    confidence: float, rule_ok: bool, reason: str):
    con = _connect()
    con.execute("""
    INSERT INTO decisions(ts,symbol,price,action,ml_signal,confidence,rule_ok,reason)
    VALUES(?,?,?,?,?,?,?,?)
    """, (_utc_now(), symbol, price, action, ml_signal, confidence, int(rule_ok), reason))
    con.commit()
    con.close()


def insert_risk_snapshot(equity: float, cash: float, buying_power: float, gross_exposure: float,
                         net_exposure: float, invested_pct: float, unrealized_pl: float,
                         realized_pl: float, drawdown_pct: float):
    con = _connect()
    con.execute("""
    INSERT INTO risk_snapshots(ts,equity,cash,buying_power,gross_exposure,net_exposure,invested_pct,
                              unrealized_pl,realized_pl,drawdown_pct)
    VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (_utc_now(), equity, cash, buying_power, gross_exposure, net_exposure, invested_pct,
          unrealized_pl, realized_pl, drawdown_pct))
    con.commit()
    con.close()


def upsert_model_health(cv_f1: float, last_train_ts: str, buy_rate: float, hold_rate: float,
                        sell_rate: float, avg_confidence: float):
    con = _connect()
    con.execute("""
    INSERT INTO model_health(id,cv_f1,last_train_ts,buy_rate,hold_rate,sell_rate,avg_confidence,updated_ts)
    VALUES(1,?,?,?,?,?,?,?)
    ON CONFLICT(id) DO UPDATE SET
      cv_f1=excluded.cv_f1,
      last_train_ts=excluded.last_train_ts,
      buy_rate=excluded.buy_rate,
      hold_rate=excluded.hold_rate,
      sell_rate=excluded.sell_rate,
      avg_confidence=excluded.avg_confidence,
      updated_ts=excluded.updated_ts;
    """, (cv_f1, last_train_ts, buy_rate, hold_rate, sell_rate, avg_confidence, _utc_now()))
    con.commit()
    con.close()


def upsert_engine_state(trading_enabled: bool, kill_switch: bool,
                        last_resync_ts: Optional[str] = None,
                        last_enforce_ts: Optional[str] = None,
                        last_enforce_action: Optional[str] = None,
                        last_universe_refresh_ts: Optional[str] = None):
    con = _connect()
    con.execute("""
    INSERT INTO engine_state(
        id, trading_enabled, kill_switch,
        last_resync_ts, last_enforce_ts, last_enforce_action,
        last_universe_refresh_ts, updated_ts
    )
    VALUES(1,?,?,?,?,?,?,?)
    ON CONFLICT(id) DO UPDATE SET
      trading_enabled=excluded.trading_enabled,
      kill_switch=excluded.kill_switch,
      last_resync_ts=COALESCE(excluded.last_resync_ts, engine_state.last_resync_ts),
      last_enforce_ts=COALESCE(excluded.last_enforce_ts, engine_state.last_enforce_ts),
      last_enforce_action=COALESCE(excluded.last_enforce_action, engine_state.last_enforce_action),
      last_universe_refresh_ts=COALESCE(excluded.last_universe_refresh_ts, engine_state.last_universe_refresh_ts),
      updated_ts=excluded.updated_ts;
    """, (
        int(trading_enabled), int(kill_switch),
        last_resync_ts, last_enforce_ts, last_enforce_action,
        last_universe_refresh_ts, _utc_now()
    ))
    con.commit()
    con.close()



def insert_order(ts: str, order_id: str, client_order_id: str, symbol: str, side: str,
                 qty: float, otype: str, tif: str, status: str, submitted_price: float,
                 filled_qty: float = 0.0, filled_avg_price: float = 0.0, error: str = ""):
    con = _connect()
    con.execute("""
    INSERT INTO orders(ts,order_id,client_order_id,symbol,side,qty,type,tif,status,submitted_price,filled_qty,filled_avg_price,error,updated_ts)
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (ts, order_id, client_order_id, symbol, side, qty, otype, tif, status, submitted_price,
          filled_qty, filled_avg_price, error, _utc_now()))
    con.commit()
    con.close()


def update_order(order_id: str, status: str, filled_qty: float, filled_avg_price: float, error: str = ""):
    con = _connect()
    con.execute("""
    UPDATE orders SET status=?, filled_qty=?, filled_avg_price=?, error=?, updated_ts=?
    WHERE order_id=?
    """, (status, filled_qty, filled_avg_price, error, _utc_now(), order_id))
    con.commit()
    con.close()


def insert_fill(order_id: str, symbol: str, side: str, qty: float, price: float):
    con = _connect()
    con.execute("""
    INSERT INTO fills(ts,order_id,symbol,side,qty,price)
    VALUES(?,?,?,?,?,?)
    """, (_utc_now(), order_id, symbol, side, qty, price))
    con.commit()
    con.close()


def upsert_position_snapshot(symbol: str, qty: float, avg_entry_price: float, current_price: float,
                             market_value: float, unrealized_pl: float, stop_price: float, trail_high: float):
    con = _connect()
    con.execute("""
    INSERT INTO positions_snapshot(symbol,qty,avg_entry_price,current_price,market_value,unrealized_pl,stop_price,trail_high,updated_ts)
    VALUES(?,?,?,?,?,?,?,?,?)
    ON CONFLICT(symbol) DO UPDATE SET
      qty=excluded.qty,
      avg_entry_price=excluded.avg_entry_price,
      current_price=excluded.current_price,
      market_value=excluded.market_value,
      unrealized_pl=excluded.unrealized_pl,
      stop_price=excluded.stop_price,
      trail_high=excluded.trail_high,
      updated_ts=excluded.updated_ts
    """, (symbol, qty, avg_entry_price, current_price, market_value, unrealized_pl, stop_price, trail_high, _utc_now()))
    con.commit()
    con.close()


# ---------- Dashboard reads ----------
def q_one(sql: str, args=()) -> Optional[dict]:
    con = _connect()
    row = con.execute(sql, args).fetchone()
    con.close()
    return dict(row) if row else None


def q_all(sql: str, args=()) -> List[dict]:
    con = _connect()
    rows = con.execute(sql, args).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_engine_state() -> Optional[dict]:
    return q_one("SELECT * FROM engine_state WHERE id=1")


def get_latest_universe() -> Optional[dict]:
    return q_one("SELECT * FROM universe ORDER BY id DESC LIMIT 1")


def get_signals() -> List[dict]:
    return q_all("SELECT * FROM signals ORDER BY symbol ASC")


def get_latest_risk() -> Optional[dict]:
    return q_one("SELECT * FROM risk_snapshots ORDER BY id DESC LIMIT 1")


def get_risk_series(limit: int = 300) -> List[dict]:
    return q_all("SELECT * FROM risk_snapshots ORDER BY id DESC LIMIT ?", (limit,))[::-1]


def get_positions() -> List[dict]:
    return q_all("SELECT * FROM positions_snapshot ORDER BY market_value DESC")


def get_orders(limit: int = 200) -> List[dict]:
    return q_all("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))


def get_decisions(limit: int = 300) -> List[dict]:
    return q_all("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))

def insert_alert(level: str, category: str, title: str, message: str):
    con = _connect()
    con.execute("""
    INSERT INTO alerts(ts, level, category, title, message)
    VALUES(?,?,?,?,?)
    """, (_utc_now(), level, category, title, message))
    con.commit()
    con.close()
