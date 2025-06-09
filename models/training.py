import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_models(data_path, report_path, model_path):
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    df.dropna(inplace=True)
    if 'Target' not in df.columns:
        raise ValueError("Missing 'Target' column")

    # Keep only valid classes
    df = df[df['Target'].isin([0, 1])]
    majority = df[df['Target'] == 0]
    minority = df[df['Target'] == 1]

    if len(minority) == 0 or len(majority) == 0:
        raise ValueError("Insufficient class variety in Target column.")

    # Upsample minority class
    minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=42)
    df = pd.concat([majority, minority_upsampled])
    df = df.sample(frac=1, random_state=42)

    # Features and labels
    X = df.drop(columns=['Target'])
    y = df['Target'].astype(int)

    # Define Random Forest model
    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_split=5, random_state=42
    )

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    with open(report_path, "w", encoding='utf-8') as f:
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            scores.append(acc)

        avg_score = np.mean(scores)
        f.write("Model: Random Forest\n")
        f.write(f" Time Series CV Accuracy: {avg_score:.4f}\n\n")

    # Retrain on full data
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model
