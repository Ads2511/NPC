# scripts/ml_models/npc_trend.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_and_predict(df: pd.DataFrame) -> float:
    df = df.copy()
    df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["ema5"] = df["close"].ewm(span=5).mean()
    df["ema10"] = df["close"].ewm(span=10).mean()
    df["momentum"] = df["close"] - df["close"].shift(10)
    df["rsi"] = 100 - (100 / (1 + df["close"].pct_change().rolling(14).mean() / df["close"].pct_change().rolling(14).std()))
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)

    features = ["rsi", "macd", "ema5", "ema10", "momentum"]
    target = "target"

    X = df[features]
    y = df[target]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    latest_scaled = scaler.transform([X.iloc[-1]])
    trend_prob = model.predict_proba(latest_scaled)[0][1]
    shift = (trend_prob - 0.5) * 0.00005  # Convert to basis point shift

    return shift
