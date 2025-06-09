import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import control
import os

# Streamlit dashboard config
st.set_page_config(page_title="SmartSignal AI", layout="centered")
st.title("SmartSignal AI")
ticker = st.text_input("Enter stock ticker:", "").upper()
try:
    stock_info = yf.Ticker(ticker).info
    valid = stock_info.get("regularMarketPrice") is not None
except Exception:
    valid = False


# Run prediction
if st.button("Predict") and ticker:
    if not valid:
        st.error("Invalid ticker symbol. Please enter a valid stock ticker.")
        st.stop()
    
    with st.spinner("Running prediction pipeline..."):
        control.control(ticker)

    # Load prediction results
    try:
        df = pd.read_csv("results/final_predictions.csv", index_col="Date", parse_dates=True)
    except FileNotFoundError:
        st.error("Prediction file not found.")
        st.stop()

    # Load trained model
    try:
        model = joblib.load("models/trained_model.pkl")
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
