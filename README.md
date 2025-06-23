# SmartSignal AI

![AI Stock Prediction](https://img.shields.io/badge/Powered%20By-Machine%20Learning-blueviolet)
![Speed](https://img.shields.io/badge/Workflow-Under%2015s-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-90%25%2B-brightgreen)

SmartSignal AI is a fast, modern stock signal system powered by machine learning. It uses a Random Forest model trained on 10+ technical indicators to generate daily buy/sell signals for any stock ticker. The workflow is fully automated, backtested, and visualized in a beautiful Streamlit dashboard.

---

## 🚀 Key Features

- **Lightning-Fast Workflow**: Full pipeline (download, indicators, training, prediction, backtest, reporting) runs in ~15 seconds per ticker.
- **90%+ Accuracy**: Achieves over 90% accuracy on recent test sets.
- **Per-Ticker Model**: Each stock gets its own custom-trained Random Forest model for maximum accuracy.
- **Smart Data Window**: Uses the last 5 years of data for a balance of speed and predictive power.
- **Modern ML**: Random Forest (160 trees, max depth 20, 3-fold CV, all CPU cores).
- **Classic Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, ATR, OBV, and more.
- **Real-Time Terminal Log**: See every step of the workflow live in the dashboard, just like a terminal.
- **Intuitive UI**: Streamlit dashboard with tabs for predictions, market data, performance, features, and reports.
- **Automatic Reporting**: Backtest results, feature importance, and summary reports saved for every run.
- **Example Data**: Example results and reports are available in the `results/` directory for quick exploration.

---

## 🛠️ Quickstart

1. **Clone the Repo**
   ```bash
   git clone https://github.com/IvanI23/SmartSignal-AI.git
   cd AI_powered_Stocks
   ```
2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the Dashboard**
   ```bash
   streamlit run dashboard.py
   ```

> **Note:** Model training and prediction for each ticker typically takes 10-20 seconds, depending on your hardware and internet speed.
> 
> **Example data and reports are available in the `results/` directory.**

---

## ⚡ How It Works

1. **Enter a Stock Ticker** in the dashboard and click Predict.
2. **Live Terminal Log** shows each step:
   - Download last 5 years of price data
   - Calculate technical indicators
   - Train a custom Random Forest model for the ticker
   - Predict buy/sell signals for every day
   - Backtest the strategy vs. Buy & Hold
   - Generate feature importance and summary reports
3. **Results Displayed Instantly**: See today's prediction, recent market data, performance charts, feature importances, and detailed reports—all in one place.

---

## 📊 For Traders & Learners
- **Perfect for:**
  - Swing traders, DIY investors, and anyone learning technical analysis or ML for finance
- **Key Points:**
  - Uses classic indicators (RSI, MACD, Bollinger Bands, etc.)
  - Simple, actionable buy/sell signals
  - Backtested, but not guaranteed—always paper trade first!

> ⚠️ **Disclaimer:** Educational use only. Not financial advice. Use at your own risk.
