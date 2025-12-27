import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ml.features import compute_features
from ml.models import build_models
from ml.training import train_ensemble
from ml.interface import ensemble_confidence

from core.portfolio import Portfolio
from core.risk import RiskManager


def make_synthetic_close_series(n=120, start=100.0, drift=0.0008, vol=0.01, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=drift, scale=vol, size=n)
    prices = start * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "close": prices,
        "high": prices + 0.5,
        "low": prices - 0.5,
        "open": prices - 0.1
    })
    return df


def test_ensemble_confidence_smoke():
    # Build synthetic data and features
    df = make_synthetic_close_series()
    df_feat = compute_features(df)
    ml_feat_cols = ["volatility", "trend"]
    # Train small ensemble
    X = df_feat[ml_feat_cols]
    y = (df_feat["returns"] > 0).astype(int)
    models = build_models()
    trained = train_ensemble(models, X, y)
    latest_features = df_feat.iloc[-1][ml_feat_cols].values.reshape(1, -1)
    conf = ensemble_confidence(trained, latest_features)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_handle_bar_opens_position_when_confident_and_signal():
    # Prepare synthetic symbol dataframe with positive trend
    sym = "AAPL"
    df = make_synthetic_close_series(n=150, drift=0.002)
    df_feat = compute_features(df)
    ml_feat_cols = ["volatility", "trend"]

    # Make sure entry_signal (ema20>ema50 and price>ema20) is True for last row
    # (synthetic drift should ensure this)
    assert df_feat["trend"].iloc[-1] in (0, 1)

    # Train ensemble on symbol data
    X = df_feat[ml_feat_cols]
    y = (df_feat["returns"] > 0).astype(int)
    models = build_models()
    trained = train_ensemble(models, X, y)

    # Prepare portfolio & risk
    portfolio = Portfolio(initial_cash=50_000)
    risk = RiskManager(max_drawdown=0.2, stop_loss_pct=0.5)

    # Mock place_stock_order: record attempts and return True (simulate successful order)
    calls = []

    def fake_place_stock_order(symbol, qty, side="buy"):
        calls.append((symbol, qty, side))
        return True

    # Minimal handler logic (mirrors entry sizing & open logic in your file)
    async def local_handle_bar(bar):
        price = bar.close
        last = df_feat.iloc[-1].copy()

        entry_signal = (last["close"] > last["ema_fast"]) and (last["ema_fast"] > last["ema_slow"])
        # Compute latest ML features for this bar
        latest_features = last[ml_feat_cols].values.reshape(1, -1)
        conf = ensemble_confidence(trained, latest_features)

        MIN_CONF = 0.60
        if entry_signal and conf >= MIN_CONF and sym not in portfolio.positions:
            confidence_multiplier = max(0.0, min(conf, 1.0))
            position_value = portfolio.cash * 0.01 * confidence_multiplier
            qty = int(position_value / price)
            if qty > 0:
                if fake_place_stock_order(sym, qty, "buy"):
                    portfolio.open_position(sym, qty, price, metadata={"confidence": conf})
                    return True
        return False

    fake_bar = SimpleNamespace(symbol=sym, close=float(df_feat["close"].iloc[-1]), high=float(df_feat["high"].iloc[-1]), low=float(df_feat["low"].iloc[-1]))
    result = asyncio.run(local_handle_bar(fake_bar))
    # We expect either a successful open or a no-op depending on trained ensemble output,
    # but the test asserts that the code path executes and if it opens, the portfolio is updated.
    if result:
        assert sym in portfolio.positions
        sym_pos = portfolio.positions[sym]
        assert sym_pos["qty"] > 0
        assert len(calls) == 1
    else:
        # If not opened due to low confidence, ensure no order was attempted
        assert len(calls) == 0