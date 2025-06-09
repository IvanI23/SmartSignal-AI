import yfinance as yf
import os
import pandas as pd

# Download historical stock data using yfinance
def download_stock_data(ticker, start, end):
    save_path = f"data/raw/{ticker}.csv"
    df = yf.download(ticker, start=start, end=end)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    return save_path
