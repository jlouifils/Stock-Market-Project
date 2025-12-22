import pandas as pd
import pytest

import stockMarket

class FakeBars:
    def __init__(self):
        idx = pd.date_range('2024-01-01', periods=30, freq='D')
        df = pd.DataFrame({'high': range(30,60), 'low': range(1,31), 'close': range(5,35)}, index=idx)
        self.df = df


def test_fetch_data_monkeypatch(monkeypatch):
    monkeypatch.setattr(stockMarket.alpaca, 'get_bars', lambda symbol, timeframe, start=None: FakeBars())
    df_dict = stockMarket.fetch_data(['AAPL'], days=5, timeout=1, retries=1)
    assert 'AAPL' in df_dict
    df = df_dict['AAPL']
    # make sure RSI and EMAs computed and symbol set
    assert 'rsi' in df.columns
    assert 'ema20' in df.columns and 'ema50' in df.columns
    assert df['symbol'].iloc[0] == 'AAPL'
