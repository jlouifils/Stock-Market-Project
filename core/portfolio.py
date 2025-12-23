from datetime import datetime

class Portfolio:
    def __init__(self, initial_cash):
        self.cash = initial_cash
        self.positions = {}
        self.trade_log = []

    def total_exposure(self, price_map):
        return sum(
            qty * price_map[symbol]
            for symbol, qty in self.positions.items()
        )

    def total_equity(self, price_map):
        return self.cash + self.total_exposure(price_map)

    def open_position(self, symbol, qty, price, metadata=None):
        self.positions[symbol] = {
            "qty": qty,
            "entry_price": price,
            "entry_time": datetime.utcnow(),
            "metadata": metadata or {}
        }

    def close_position(self, symbol, price, reason):
        pos = self.positions.pop(symbol)
        pnl = pos["qty"] * (price - pos["entry_price"])

        self.cash += pos["qty"] * price

        self.trade_log.append({
            "symbol": symbol,
            "entry_price": pos["entry_price"],
            "exit_price": price,
            "qty": pos["qty"],
            "pnl": pnl,
            "reason": reason,
            "entry_time": pos["entry_time"],
            "exit_time": datetime.utcnow(),
            "metadata": pos["metadata"]
        })

        return pnl
