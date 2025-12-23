import logging
from datetime import datetime

class RiskManager:
    def __init__(
        self,
        max_drawdown=0.10,
        stop_loss_pct=0.03,
        max_position_pct=0.20,
        max_portfolio_exposure=0.80
    ):
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.max_position_pct = max_position_pct
        self.max_portfolio_exposure = max_portfolio_exposure

        self.equity_curve = []
        self.peak_equity = None
        self.trading_halted = False

    def update_equity(self, equity):
        self.equity_curve.append(equity)
        if self.peak_equity is None:
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def check_drawdown(self):
        if not self.equity_curve:
            return False

        current = self.equity_curve[-1]
        drawdown = (self.peak_equity - current) / self.peak_equity

        if drawdown >= self.max_drawdown:
            self.trading_halted = True
            logging.critical(
                f"🛑 MAX DRAWDOWN HIT ({drawdown:.2%}) — TRADING HALTED"
            )
            return True
        return False

    def check_stop_loss(self, entry_price, current_price):
        loss_pct = (entry_price - current_price) / entry_price
        return loss_pct >= self.stop_loss_pct

    def position_size_allowed(self, position_value, total_equity):
        return position_value <= total_equity * self.max_position_pct

    def portfolio_exposure_allowed(self, total_exposure, total_equity):
        return total_exposure <= total_equity * self.max_portfolio_exposure
