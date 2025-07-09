
# NPC Strategy Summary

## Strategic Approach

Our strategy utilizes a hybrid machine learning-enhanced PMM (Pure Market Making) system. It combines real-time feature engineering from live market data with lightweight predictive models trained on historical candle data. The trading decisions are made based on:

- **Volatility Prediction**: Adjust bid-ask spreads dynamically using an XGBoost regression model trained on standard deviation, Bollinger Band width, and volume data.
- **Trend Detection**: Use logistic regression on technical indicators such as RSI, MACD, and EMAs to determine short-term directional bias.
- **Inventory Skewing**: Apply linear regression to the current portfolio inventory ratio to balance risk and capital allocation.

## Assumptions and Tradeoffs

- **Candles Availability**: We assume that reliable candle data is available in real-time from the exchange.
- **Model Simplicity**: We opt for lightweight models (e.g., XGBoost, Logistic Regression) to maintain low latency and on-the-fly re-training.
- **One-Tick Training**: Models are trained only once per session to avoid frequent overhead, assuming recent data reflects current conditions.
- **Minimal Data Drift Handling**: Retraining frequency is not adaptive; models do not continuously evolve in real-time.

## Key Risk Management Principles

- **Volatility-Based Spread Control**: Wider spreads in high volatility scenarios reduce execution risk.
- **Inventory Exposure Control**: Balancing quote and base exposure via inventory-based price skewing prevents overexposure to a single asset.
- **Order Refresh Cycle**: Orders are canceled and recreated every 15 seconds to ensure prices remain relevant and competitive.
- **Budget Enforcement**: All orders go through Hummingbot's built-in budget checker to avoid over-committing capital.

This strategy is optimized for clarity, modular design, and compatibility with Hummingbot's scripting framework.
