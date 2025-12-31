# execution.py
import logging

def place_manual_order(alpaca, symbol, qty, side):
    try:
        if qty <= 0:
            raise ValueError("Quantity must be positive")

        alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )

        logging.info(f"🖱️ MANUAL {side.upper()} | {symbol} x{qty}")
        return True, f"{side.upper()} {symbol} x{qty} submitted"

    except Exception as e:
        logging.error(f"Manual order failed: {e}")
        return False, str(e)
