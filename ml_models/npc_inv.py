# scripts/ml_models/npc_inv.py

import numpy as np
from sklearn.linear_model import LinearRegression


def train_and_predict(inventory_ratio: float) -> float:
    # Simulated training: imbalance vs corrective shift
    X = np.linspace(0.4, 0.6, 100).reshape(-1, 1)
    y = (0.5 - X).ravel() * 2  # e.g. inventory skew from mid

    model = LinearRegression()
    model.fit(X, y)

    shift = model.predict(np.array([[inventory_ratio]]))[0] * 0.00005  # scale to basis points
    return shift
