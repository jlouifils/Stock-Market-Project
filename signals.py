# signals.py
from __future__ import annotations
from typing import Tuple, Union
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

BUY = "BUY"
SELL = "SELL"   # long-only meaning: EXIT / AVOID (not short)
HOLD = "HOLD"

MIN_BUY_CONF = 0.60
MIN_SELL_CONF = 0.65


def _as_2d(X) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def ml_predict(ml_model, features: Union[np.ndarray, "pd.DataFrame"]) -> Tuple[str, float, float]:
    if pd is not None and hasattr(features, "columns"):
        X = features
    else:
        X = _as_2d(features)

    proba = ml_model.predict_proba(X)[0]
    p0 = float(proba[0])
    p1 = float(proba[1]) if len(proba) > 1 else 1.0 - p0

    pred = int(ml_model.predict(X)[0])

    if pred == 1 and p1 >= MIN_BUY_CONF:
        return BUY, p1, p0
    if pred == 0 and p0 >= MIN_SELL_CONF:
        return SELL, p1, p0
    return HOLD, p1, p0


def rule_gate(price: float, ema20: float, ema50: float, rsi: float,
              ml_signal: str, p_buy: float, p_sell: float):
    if ml_signal == BUY:
        if not (price > ema20 > ema50):
            return False, "BUY blocked: EMA trend misalignment"
        if rsi > 75:
            return False, "BUY blocked: RSI too hot"
        return True, f"BUY approved (p_buy={p_buy:.2f})"

    if ml_signal == SELL:
        if price < ema20:
            return True, f"EXIT approved: price < EMA20 (p_sell={p_sell:.2f})"
        if ema20 < ema50:
            return True, f"EXIT approved: EMA20 < EMA50 (p_sell={p_sell:.2f})"
        return False, f"Risk-off warning (p_sell={p_sell:.2f}) but trend not broken"

    return False, "No actionable ML signal"
