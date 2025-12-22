import stockMarket
import pandas as pd

class FakeBars:
    def __init__(self):
        idx = pd.date_range('2024-01-01', periods=60, freq='D')
        df = pd.DataFrame({'high': range(60,120), 'low': range(1,61), 'close': range(30,90)}, index=idx)
        self.df = df

stockMarket.alpaca.get_bars = lambda symbol, timeframe, start=None: FakeBars()
print('Calling fetch_data...')
df_dict = stockMarket.fetch_data(['AAPL'], days=30, timeout=2, retries=1)
print('Got keys:', list(df_dict.keys()))
print('First rows:\n', df_dict['AAPL'].head())
print('Calling train_ml...')
model = stockMarket.train_ml(df_dict)
print('Model type:', type(model))
