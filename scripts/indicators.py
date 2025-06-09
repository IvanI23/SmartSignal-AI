import pandas as pd
import numpy as np
import warnings

# Suppress performance warnings from pandas
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Indicator calculations
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic_oscillator(high, low, close, window=14):
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    stoch = 100 * (close - lowest_low) / (highest_high - lowest_low)
    return stoch

def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def calculate_macd(close, fast=12, slow=26):
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    macd_line = ema_fast - ema_slow
    return macd_line

def calculate_bollinger_bands(close, window=20, n_std=2):
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper_band = sma + (std * n_std)
    lower_band = sma - (std * n_std)
    return upper_band, lower_band

def calculate_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window).mean()
    return atr

def calculate_obv(close, volume):
    direction = np.where(close.diff() > 0, 1, np.where(close.diff() < 0, -1, 0))
    obv = (direction * volume).cumsum()
    return pd.Series(obv, index=close.index)

def calculate_volume_sma(volume, window=20):
    return volume.rolling(window).mean()

def calculate_log_returns(close):
    return np.log(close).diff()

def validate_and_clean(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    if 'RSI' in df.columns:
        df['RSI'] = df['RSI'].clip(0, 100)
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].round(4)
    return df


# Indicator API
def add_indicators(file, ticker):
    save_path = f"data/processed/{ticker}_indicators.csv"

    df = pd.read_csv(
        file,
        skiprows=3,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
        header=None,
        parse_dates=["Date"],
        index_col="Date"
    )
    df.dropna(inplace=True)

    try:

        df['RSI'] = calculate_rsi(df['Close'])
        df['Stoch'] = calculate_stochastic_oscillator(df['High'], df['Low'], df['Close'])


        df['EMA_20'] = calculate_ema(df['Close'], 20)
        df['MACD'] = calculate_macd(df['Close'])


        df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df['Close'])
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])


        df['OBV'] = calculate_obv(df['Close'], df['Volume'])
        df['Volume_SMA_20'] = calculate_volume_sma(df['Volume'])


        df['Log_Returns'] = calculate_log_returns(df['Close'])

        df['Target'] = (df['Close'].shift(-3) > df['Close'] * 1.01).astype(int)

        df = validate_and_clean(df)

    except Exception as e:
        raise ValueError(f"Indicator calculation failed: {e}")

    df.to_csv(save_path)
    return df
