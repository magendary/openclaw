# BTCUSDT 5-minute candle V1 backtest — 2025-12-31 UTC

## Scope

V1 deliberately uses only two inputs:

1. elapsed seconds in the active five-minute candle;
2. current price distance from that candle's open, measured in basis points.

The probability model is an empirical lookup table with Beta smoothing and local-bin expansion. No RSI, MACD, volume, order book, Polymarket price, or machine-learning classifier is used.

## Data split

- Source: Binance Spot `BTCUSDT` one-minute klines.
- Training: 2025-01-01 through 2025-12-30 UTC.
- Test: 2025-12-31 UTC only.
- Training five-minute windows: 104,832.
- Test five-minute windows: 288.
- Snapshots per window: 60, 120, 180, and 240 seconds.
- Distance-bin width: 1 bp.
- Minimum local sample count: 200.
- Beta prior: mean 0.50, strength 20.

## Direction results

| Snapshot | Predictions | V1 accuracy | Current-color baseline | Brier score | Mean confidence |
|---:|---:|---:|---:|---:|---:|
| 60s | 288 | 67.36% | 67.36% | 0.2114 | 63.22% |
| 120s | 288 | 74.31% | 74.31% | 0.1768 | 69.88% |
| 180s | 288 | 84.72% | 84.72% | 0.1262 | 76.46% |
| 240s | 288 | 90.28% | 90.28% | 0.0824 | 84.07% |
| All | 1,152 | 79.17% | 79.17% | 0.1492 | 73.41% |

## Dynamic first-signal results

For each five-minute candle, select the first 60/120/180/240-second snapshot whose confidence reaches the threshold. Candles that never reach it are skipped.

| Confidence threshold | Signals | Coverage | Accuracy | Mean entry second | Mean confidence |
|---:|---:|---:|---:|---:|---:|
| 55.00% | 288 | 100.00% | 70.49% | 70.8s | 64.78% |
| 60.00% | 288 | 100.00% | 74.65% | 85.8s | 67.70% |
| 65.00% | 274 | 95.14% | 83.21% | 120.9s | 73.56% |
| 70.00% | 261 | 90.62% | 86.59% | 141.8s | 77.30% |
| 75.00% | 221 | 76.74% | 91.86% | 162.1s | 82.24% |
| 80.00% | 209 | 72.57% | 93.78% | 191.8s | 86.14% |
| 85.00% | 167 | 57.99% | 96.41% | 204.4s | 90.09% |
| 90.00% | 142 | 49.31% | 97.89% | 221.8s | 92.84% |
| 95.00% | 60 | 20.83% | 100.00% | 237.0s | 96.36% |

## Probability calibration

| Predicted P(Up) bucket | Predictions | Mean predicted P(Up) | Actual Up rate | Absolute error |
|---|---:|---:|---:|---:|
| (-0.001, 0.1] | 104 | 6.68% | 2.88% | 3.79% |
| (0.1, 0.2] | 94 | 15.79% | 7.45% | 8.34% |
| (0.2, 0.3] | 147 | 25.26% | 21.09% | 4.17% |
| (0.3, 0.4] | 166 | 35.61% | 31.33% | 4.28% |
| (0.4, 0.5] | 75 | 43.93% | 40.00% | 3.93% |
| (0.5, 0.6] | 86 | 55.46% | 51.16% | 4.29% |
| (0.6, 0.7] | 188 | 64.07% | 73.94% | 9.87% |
| (0.7, 0.8] | 122 | 74.99% | 85.25% | 10.25% |
| (0.8, 0.9] | 90 | 84.40% | 91.11% | 6.71% |
| (0.9, 1.0] | 80 | 93.79% | 100.00% | 6.21% |

## Interpretation limits

- This is a one-day directional backtest, not evidence of durable profitability.
- It does not include Polymarket bid/ask prices, fees, slippage, or Chainlink settlement differences.
- One-minute inputs mean the earliest dynamic update is at 60 seconds. A later version can move to one-second data without changing the model definition.
- The test day was not used to fit the probability table.
