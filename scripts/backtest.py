import pandas as pd
import matplotlib.pyplot as plt

def run_backtest(data, predictions, initial_cash=10000, trading_fee=0.001):
    df = data.copy()
    df['Signal'] = predictions
    df['Signal'] = df['Signal'].fillna(0)

    # Portfolio initial states
    cash = initial_cash
    holding = 0
    position = 0  
    portfolio_values = []

    for date, row in df.iterrows():
        price = row['Close']
        signal = row['Signal']

        # Buy
        if signal == 1 and position == 0:
            shares_to_buy = cash // price
            cost = shares_to_buy * price * (1 + trading_fee)
            if cost <= cash:
                holding = shares_to_buy
                cash -= cost
                position = 1 

        # Sell
        elif signal == 0 and position == 1:
            proceeds = holding * price * (1 - trading_fee)
            cash += proceeds
            holding = 0
            position = 0 

        # Portfolio value
        portfolio_value = cash + holding * price
        portfolio_values.append(portfolio_value)

    df['Model_Strategy'] = portfolio_values

    # Buy and hold Simulation
    first_price = df['Close'].iloc[0]
    bh_shares = initial_cash // first_price
    bh_cash = initial_cash - (bh_shares * first_price)
    df['Buy_Hold'] = df['Close'] * bh_shares + bh_cash

    return df[['Model_Strategy', 'Buy_Hold', 'Signal']]

# Plot
def plot_backtest(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Model_Strategy'], label='Strategy', linewidth=2)
    plt.plot(df.index, df['Buy_Hold'], label='Buy & Hold', color = 'red', linewidth=2)
    plt.title("Portfolio Growth")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/backtest_plot.png")

