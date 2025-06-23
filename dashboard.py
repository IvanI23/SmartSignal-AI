import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import os
import time
from scripts.logger import WorkflowLogger
from scripts.downloader import download_stock_data
from scripts.indicators import add_indicators
from models.training import train_models
from scripts.backtest import run_backtest, plot_backtest
from scripts.reporting import plot_feature_importance, write_summary

# Streamlit dashboard config
st.set_page_config(page_title="SmartSignal AI", layout="centered")
st.title("SmartSignal AI")
ticker = st.text_input("Enter stock ticker:", "").upper()

# Validate ticker symbol
def is_valid_ticker(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return not data.empty
    except Exception:
        return False

valid = is_valid_ticker(ticker)

# Run prediction
if st.button("Predict") and ticker:
    if not valid:
        st.error("Invalid ticker symbol. Please enter a valid stock ticker.")
        st.stop()

    logger = WorkflowLogger()
    log_placeholder = st.empty()
    start_time = time.time()

    # Step 1: Download data
    logger.log(f"[{time.strftime('%H:%M:%S')}] Downloading data for {ticker}...")
    log_placeholder.code(logger.get_log(), language="bash")
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    raw_path = f"data/raw/{ticker}.csv"
    processed_path = f"data/processed/{ticker}_indicators.csv"
    model_path = f"models/{ticker}_model.pkl"
    save_path = download_stock_data(ticker, "2015-01-01", today)

    # Step 2: Calculate indicators
    logger.log(f"[{time.strftime('%H:%M:%S')}] Calculating indicators for {ticker}...")
    log_placeholder.code(logger.get_log(), language="bash")
    add_indicators(save_path, ticker)

    # Step 3: Train model
    logger.log(f"[{time.strftime('%H:%M:%S')}] Training model for {ticker}...")
    log_placeholder.code(logger.get_log(), language="bash")
    model = train_models(
        data_path=processed_path,
        report_path="results/model_reports.txt",
        model_path=model_path
    )

    # Step 4: Make predictions
    logger.log(f"[{time.strftime('%H:%M:%S')}] Making predictions...")
    log_placeholder.code(logger.get_log(), language="bash")
    df = pd.read_csv(processed_path, index_col="Date", parse_dates=True, low_memory=False)
    X = df.drop(columns=["Target"])
    df["Prediction"] = model.predict(X)

    # Step 5: Backtest
    logger.log(f"[{time.strftime('%H:%M:%S')}] Running backtest...")
    log_placeholder.code(logger.get_log(), language="bash")
    backtest_results = run_backtest(df, df["Prediction"])
    backtest_results.to_csv("results/backtest_results.csv")
    plot_backtest(backtest_results)

    # Step 6: Reporting
    logger.log(f"[{time.strftime('%H:%M:%S')}] Generating reports...")
    log_placeholder.code(logger.get_log(), language="bash")
    final_value = backtest_results['Model_Strategy'].iloc[-1]
    buy_hold_value = backtest_results['Buy_Hold'].iloc[-1]
    df["Signal"] = df["Prediction"]
    df["Correct"] = (df["Signal"] == df["Target"]).astype(int)
    rolling_accuracy = df["Correct"].rolling(20).mean().iloc[-1]
    plot_feature_importance(model, X)
    write_summary(final_value, buy_hold_value, accuracy=rolling_accuracy)
    df.to_csv("results/final_predictions.csv")

    logger.log(f"[{time.strftime('%H:%M:%S')}] Workflow complete!")
    log_placeholder.code(logger.get_log(), language="bash")
    elapsed_time = time.time() - start_time
    time.sleep(0.5)
    log_placeholder.empty()
    st.success(f"✅ Prediction completed in {elapsed_time:.2f} seconds.")

    # Load prediction results
    try:
        df = pd.read_csv("results/final_predictions.csv", index_col="Date", parse_dates=True)
    except FileNotFoundError:
        st.error("Prediction file not found.")
        st.stop()

    # Load trained model
    try:
        model = joblib.load(f"models/{ticker}_model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Prepare features for prediction
    X_all = df.drop(columns=["Target", "Prediction", "Signal", "Correct", "Action"], errors='ignore')
    if len(X_all) == 0:
        st.warning("No features available for prediction.")
        st.stop()

    latest_features = X_all.iloc[[-1]]
    today_pred = int(model.predict(latest_features)[0])
    label = "📈 Buy" if today_pred == 1 else "📉 Sell"
    today_date = df.index[-1] + pd.Timedelta(days=1)

    # Add predicted action to dataframe for display
    df["Action"] = df.get("Prediction", df.get("Signal", 0)).apply(lambda x: "Buy" if x == 1 else "Sell")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚡ Today's Prediction", 
        "📊 Market Data", 
        "📈 Performance", 
        "📌 Features", 
        "📑 Reports"
    ])

    # Tab 1: Today's Prediction
    with tab1:
        st.subheader("Today's Suggested Move")
        st.markdown(f"**Date:** `{today_date.date()}`")
        st.markdown(f"**Prediction:** {label}")

    # Tab 2: Market Data
    with tab2:
        st.subheader("📊 Recent Market Data")
        st.dataframe(df[["Close", "Action"]].tail(10).iloc[::-1])
        st.subheader("📈 Closing Price Chart")
        st.line_chart(df["Close"])

    # Tab 3: Performance
    with tab3:
        st.subheader("📈 Model Backtest Performance")
        backtest_path = "results/backtest_results.csv"
        if os.path.exists(backtest_path):
            df2 = pd.read_csv(backtest_path, index_col="Date", parse_dates=True)
            st.line_chart(df2[["Model_Strategy", "Buy_Hold"]])
            st.markdown("**Our Model Strategy vs Buy & Hold Strategy**")
        else:
            st.warning("Backtest results not found.")

    # Tab 4: Feature Importance
    with tab4:
        st.subheader("📌 Feature Importance")
        feature_importance_path = "results/feature_importance.png"
        if os.path.exists(feature_importance_path):
            st.image(feature_importance_path, caption="Top Feature Importances")
        else:
            st.warning("Feature importance plot not found.")

    # Tab 5: Reports
    with tab5:
        st.subheader("📄 Summary Report")
        summary_path = "results/summary.txt"
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                st.code(f.read(), language="text")
        else:
            st.warning("Summary report file not found.")

        st.subheader("📑 Model Report")
        report_path = "results/model_reports.txt"
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                st.code(f.read(), language="text")
        else:
            st.warning("Model report file not found.")
