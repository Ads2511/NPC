# scripts/ml_models/npc_vol.py

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_and_predict(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["rolling_std"] = df["return"].rolling(10).std()
    df["bb_width"] = df["close"].rolling(20).std() * 2
    df["volume"] = df["volume"].fillna(method="ffill").fillna(0)
    df.dropna(inplace=True)

    features = ["rolling_std", "bb_width", "volume"]
    target = "rolling_std"

    X = df[features]
    y = df[target]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = XGBRegressor(n_estimators=50, max_depth=3)
    model.fit(X_train_scaled, y_train)

    X_latest_scaled = scaler.transform([X.iloc[-1]])
    pred_std = model.predict(X_latest_scaled)[0]

    spread = max(0.0001, pred_std * 0.5)
    return {"bid_spread": spread, "ask_spread": spread}
