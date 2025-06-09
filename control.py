from datetime import datetime
import pandas as pd
from scripts.indicators import add_indicators
from scripts.downloader import download_stock_data
from models.training import train_models
from scripts.backtest import run_backtest, plot_backtest
from scripts.reporting import plot_feature_importance, write_summary

# Main control function to run the entire pipeline
def control(ticker):
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Running control for {ticker} on {today}")

    save_path = download_stock_data(ticker, "2015-01-01", today)
    add_indicators(save_path, ticker)

    model = train_models(
        data_path=f"data/processed/{ticker}_indicators.csv",
        report_path="results/model_reports.txt",
        model_path="models/trained_model.pkl"
    )

    df = pd.read_csv(f"data/processed/{ticker}_indicators.csv", index_col="Date", parse_dates=True)
    X = df.drop(columns=["Target"])
    df["Prediction"] = model.predict(X)

    backtest_results = run_backtest(df, df["Prediction"])
    backtest_results.to_csv("results/backtest_results.csv")
    plot_backtest(backtest_results)

    final_value = backtest_results['Model_Strategy'].iloc[-1]
    buy_hold_value = backtest_results['Buy_Hold'].iloc[-1]

    df["Signal"] = df["Prediction"]
    df["Correct"] = (df["Signal"] == df["Target"]).astype(int)
    rolling_accuracy = df["Correct"].rolling(20).mean().iloc[-1]

    plot_feature_importance(model, X)
    write_summary(final_value, buy_hold_value, accuracy=rolling_accuracy)

    df.to_csv("results/final_predictions.csv")
