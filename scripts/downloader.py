import yfinance as yf
import os
from datetime import datetime, timedelta

# Download historical stock data using yfinance
def download_stock_data(ticker, start=None, end=None):
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    if start is None:
        start = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    save_path = f"data/raw/{ticker}.csv"
    df = yf.download(ticker, start=start, end=end)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    return save_path
