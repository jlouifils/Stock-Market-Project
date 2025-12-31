# alerts.py
from state_store import insert_alert

def alert_signal(symbol: str, signal: str, confidence: float, rule_ok: bool, reason: str):
    title = f"{signal} candidate: {symbol}"
    msg = f"conf={confidence:.2f} rule_ok={rule_ok} reason={reason}"
    insert_alert("INFO", "SIGNAL", title, msg)

def alert_risk(title: str, message: str, level: str = "WARN"):
    insert_alert(level, "RISK", title, message)

def alert_system(title: str, message: str, level: str = "ERROR"):
    insert_alert(level, "SYSTEM", title, message)
