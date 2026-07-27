# Candlestick V1

A deliberately simple BTC five-minute candle direction model.

## V1 hypothesis

At any observation point inside a five-minute candle, estimate:

```text
P(final five-minute close >= five-minute open)
```

using only:

1. elapsed seconds in the current candle;
2. current price distance from the candle open in basis points.

No technical indicators, volume, order book, external market probabilities, or complex classifier are used.

## Method

- Download Binance Spot `BTCUSDT` one-minute klines from Binance Vision.
- Reconstruct UTC-aligned five-minute windows.
- Create snapshots at 60, 120, 180, and 240 seconds.
- Fit an empirical two-dimensional probability table on historical windows.
- Use Beta smoothing and neighboring distance bins when an exact cell is sparse.
- Keep the selected test day fully out of training.
- Compare V1 against the baseline: current candle color stays unchanged until close.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
candlestick-backtest \
  --train-start 2025-01-01 \
  --test-date 2025-12-31 \
  --output-dir reports/2025-12-31
```

The first run downloads monthly archives into `.cache/binance/`. Generated outputs include:

- `REPORT.md`
- `summary.json`
- `predictions.csv`
- `probability_table.csv`
- `dynamic_thresholds.csv`
- `calibration.csv`

## What this backtest does not claim

This is a directional Binance-price baseline. It does not yet model Polymarket execution prices, fees, slippage, liquidity, or Chainlink-vs-Binance settlement differences. Those belong in later versions after the clean V1 baseline is established.
