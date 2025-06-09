import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_models(data_path, report_path, model_path):
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    df.dropna(inplace=True)
    if 'Target' not in df.columns:
        raise ValueError("Missing 'Target' column")

    X = df.drop(columns=['Target'])
    y = df['Target']

    # Balance the dataset
    df['Target'] = y
    df = df[df['Target'].isin([0, 1])] 
    majority = df[df['Target'] == 0]
    minority = df[df['Target'] == 1]

    if len(minority) == 0 or len(majority) == 0:
        raise ValueError("Insufficient class variety in Target column.")

    # Upsample minority class
    # Logic - Will ensure exposure to rare positive cases, shuffling helps in generalization
    minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=42)
    df = pd.concat([majority, minority_upsampled])
    df = df.sample(frac=1, random_state=42)

    # Feature and target separation
    X = df.drop(columns=['Target'])
    y = df['Target'].astype(int)


    # Two models to get optimised solutions
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # Train on past, test on future
    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_score = 0
    best_model_name = ""

    # Train and evaluate modelsq
    with open(report_path, "w", encoding='utf-8') as f:
        for name, model in models.items():
            scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                scores.append(acc)

            avg_score = np.mean(scores)
            f.write(f"Model: {name}\n")
            f.write(f" Time Series CV Accuracy: {avg_score:.4f}\n\n")

            if avg_score > best_score:
                best_score = avg_score
                best_model = model
                best_model_name = name

        f.write(f"Model Used: {best_model_name}\n")

    best_model.fit(X, y)  
    joblib.dump(best_model, model_path)
    print(f" Model Used: {best_model_name}")
    print(f" Saved to: {model_path}")
    return best_model
