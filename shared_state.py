# shared_state.py
from collections import defaultdict
from threading import Lock

lock = Lock()

market_state = defaultdict(lambda: {
    "price": None,
    "ema20": None,
    "ema50": None,
    "ml_signal": None,   # BUY / SELL / HOLD
    "confidence": None,
    "rule_ok": None,     # True / False
    "reason": None       # why blocked or approved
})
