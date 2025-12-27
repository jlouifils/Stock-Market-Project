import threading

lock = threading.Lock()

prices = {}
positions = {}
signals = []      # buy/sell markers
equity_curve = []
pnl = 0.0
