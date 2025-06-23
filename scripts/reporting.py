import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Importance of indicators set by tree-based models
def plot_feature_importance(model, X):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.gca().invert_yaxis()
        plt.title("Top Feature Importances")
        plt.tight_layout()

        plt.savefig("results/feature_importance.png")
        plt.close()
    else:
        print("Model does not support feature importance.")

# Important summary of the strategy performance
def write_summary(final_value, buy_hold_value, accuracy, output_path="results/summary.txt"):
    strategy_return = (final_value - 10000) / 10000
    buy_hold_return = (buy_hold_value - 10000) / 10000
    today = datetime.now().strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"FINAL STRATEGY SUMMARY ({today})\n")
        f.write(f"Strategy Final Value: ${final_value:.2f}\n")
        f.write(f"Buy & Hold Final Value: ${buy_hold_value:.2f}\n")
        f.write(f"Strategy Return: {strategy_return*100:.2f}%\n")
        f.write(f"Buy & Hold Return: {buy_hold_return*100:.2f}%\n")
        f.write(f"Model Accuracy (on recent test set): {accuracy*100:.2f}%\n")
