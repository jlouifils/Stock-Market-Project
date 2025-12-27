class Portfolio:
    def __init__(self, cash: float):
        self.cash = cash
        self.positions = {}

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def add_position(self, symbol: str, qty: int, entry: float):
        self.positions[symbol] = {
            "qty": qty,
            "entry": entry
        }

    def remove_position(self, symbol: str):
        if symbol in self.positions:
            del self.positions[symbol]
