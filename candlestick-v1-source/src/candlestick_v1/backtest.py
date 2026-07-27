from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_five_minute_snapshots, load_klines, split_snapshots
from .model import ProbabilityTableModel


@dataclass(frozen=True)
class MetricRow:
    elapsed_seconds: int | str
    predictions: int
    model_accuracy: float
    current_color_accuracy: float
    brier_score: float
    mean_confidence: float


def _accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    return float((actual.to_numpy() == predicted.to_numpy()).mean())


def _brier(actual: pd.Series, probability: pd.Series) -> float:
    return float(np.mean((probability.to_numpy() - actual.to_numpy()) ** 2))


def compute_metrics(predictions: pd.DataFrame) -> list[MetricRow]:
    frame = predictions.copy()
    frame["current_color_up"] = (frame["distance_from_open_bps"] >= 0).astype(int)
    rows: list[MetricRow] = []

    def make_row(label: int | str, group: pd.DataFrame) -> MetricRow:
        return MetricRow(
            elapsed_seconds=label,
            predictions=len(group),
            model_accuracy=_accuracy(group["final_up"], group["predicted_up"]),
            current_color_accuracy=_accuracy(group["final_up"], group["current_color_up"]),
            brier_score=_brier(group["final_up"], group["p_up"]),
            mean_confidence=float(group["confidence"].mean()),
        )

    for elapsed, group in frame.groupby("elapsed_seconds", sort=True):
        rows.append(make_row(int(elapsed), group))
    rows.append(make_row("all", frame))
    return rows


def dynamic_threshold_metrics(
    predictions: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    total_windows = int(predictions["window_start"].nunique())
    for threshold in thresholds:
        eligible = predictions[predictions["confidence"] >= threshold]
        selected = (
            eligible.sort_values(["window_start", "elapsed_seconds"])
            .groupby("window_start", as_index=False)
            .first()
        )
        if selected.empty:
            rows.append(
                {
                    "threshold": threshold,
                    "trades": 0,
                    "coverage": 0.0,
                    "accuracy": math.nan,
                    "mean_entry_second": math.nan,
                    "mean_confidence": math.nan,
                }
            )
            continue
        rows.append(
            {
                "threshold": threshold,
                "trades": len(selected),
                "coverage": len(selected) / total_windows,
                "accuracy": _accuracy(selected["final_up"], selected["predicted_up"]),
                "mean_entry_second": float(selected["elapsed_seconds"].mean()),
                "mean_confidence": float(selected["confidence"].mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, 11)
    frame = predictions.copy()
    frame["probability_bucket"] = pd.cut(
        frame["p_up"], bins=bins, include_lowest=True, right=True
    )
    grouped = (
        frame.groupby("probability_bucket", observed=True)
        .agg(
            predictions=("final_up", "size"),
            mean_predicted_p_up=("p_up", "mean"),
            actual_up_rate=("final_up", "mean"),
        )
        .reset_index()
    )
    grouped["absolute_calibration_error"] = (
        grouped["mean_predicted_p_up"] - grouped["actual_up_rate"]
    ).abs()
    grouped["probability_bucket"] = grouped["probability_bucket"].astype(str)
    return grouped


def _format_percent(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.2%}"


def render_report(
    *,
    symbol: str,
    train_start: date,
    test_date: date,
    train_windows: int,
    test_windows: int,
    metrics: list[MetricRow],
    dynamic: pd.DataFrame,
    calibration: pd.DataFrame,
    model: ProbabilityTableModel,
) -> str:
    lines = [
        f"# {symbol} 5-minute candle V1 backtest — {test_date.isoformat()} UTC",
        "",
        "## Scope",
        "",
        "V1 deliberately uses only two inputs:",
        "",
        "1. elapsed seconds in the active five-minute candle;",
        "2. current price distance from that candle's open, measured in basis points.",
        "",
        "The probability model is an empirical lookup table with Beta smoothing and local-bin expansion. No RSI, MACD, volume, order book, Polymarket price, or machine-learning classifier is used.",
        "",
        "## Data split",
        "",
        f"- Source: Binance Spot `{symbol}` one-minute klines.",
        f"- Training: {train_start.isoformat()} through {(test_date - timedelta(days=1)).isoformat()} UTC.",
        f"- Test: {test_date.isoformat()} UTC only.",
        f"- Training five-minute windows: {train_windows:,}.",
        f"- Test five-minute windows: {test_windows:,}.",
        f"- Snapshots per window: 60, 120, 180, and 240 seconds.",
        f"- Distance-bin width: {model.bin_width_bps:g} bp.",
        f"- Minimum local sample count: {model.min_samples:,}.",
        f"- Beta prior: mean {model.prior_mean:.2f}, strength {model.prior_strength:g}.",
        "",
        "## Direction results",
        "",
        "| Snapshot | Predictions | V1 accuracy | Current-color baseline | Brier score | Mean confidence |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        snapshot = "All" if row.elapsed_seconds == "all" else f"{row.elapsed_seconds}s"
        lines.append(
            "| "
            + " | ".join(
                [
                    snapshot,
                    f"{row.predictions:,}",
                    _format_percent(row.model_accuracy),
                    _format_percent(row.current_color_accuracy),
                    f"{row.brier_score:.4f}",
                    _format_percent(row.mean_confidence),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Dynamic first-signal results",
            "",
            "For each five-minute candle, select the first 60/120/180/240-second snapshot whose confidence reaches the threshold. Candles that never reach it are skipped.",
            "",
            "| Confidence threshold | Signals | Coverage | Accuracy | Mean entry second | Mean confidence |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dynamic.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_percent(float(row.threshold)),
                    f"{int(row.trades):,}",
                    _format_percent(float(row.coverage)),
                    _format_percent(float(row.accuracy)),
                    "—" if pd.isna(row.mean_entry_second) else f"{float(row.mean_entry_second):.1f}s",
                    _format_percent(float(row.mean_confidence)),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Probability calibration",
            "",
            "| Predicted P(Up) bucket | Predictions | Mean predicted P(Up) | Actual Up rate | Absolute error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in calibration.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.probability_bucket),
                    f"{int(row.predictions):,}",
                    _format_percent(float(row.mean_predicted_p_up)),
                    _format_percent(float(row.actual_up_rate)),
                    _format_percent(float(row.absolute_calibration_error)),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This is a one-day directional backtest, not evidence of durable profitability.",
            "- It does not include Polymarket bid/ask prices, fees, slippage, or Chainlink settlement differences.",
            "- One-minute inputs mean the earliest dynamic update is at 60 seconds. A later version can move to one-second data without changing the model definition.",
            "- The test day was not used to fit the probability table.",
            "",
        ]
    )
    return "\n".join(lines)


def run_backtest(
    *,
    symbol: str,
    train_start: date,
    test_date: date,
    cache_dir: Path,
    output_dir: Path,
    bin_width_bps: float,
    min_samples: int,
    prior_strength: float,
) -> dict[str, object]:
    klines = load_klines(
        symbol=symbol,
        interval="1m",
        start_date=train_start,
        end_date=test_date,
        cache_dir=cache_dir,
    )
    snapshots = build_five_minute_snapshots(klines)
    split = split_snapshots(snapshots, test_date)
    if split.test_windows != 288:
        raise RuntimeError(
            f"Expected 288 complete five-minute test windows, found {split.test_windows}"
        )

    model = ProbabilityTableModel(
        bin_width_bps=bin_width_bps,
        min_samples=min_samples,
        prior_strength=prior_strength,
    ).fit(split.train)
    predictions = model.predict(split.test)
    metrics = compute_metrics(predictions)
    dynamic = dynamic_threshold_metrics(
        predictions,
        thresholds=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    )
    calibration = calibration_table(predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    model.probability_table().to_csv(output_dir / "probability_table.csv", index=False)
    dynamic.to_csv(output_dir / "dynamic_thresholds.csv", index=False)
    calibration.to_csv(output_dir / "calibration.csv", index=False)

    report = render_report(
        symbol=symbol,
        train_start=train_start,
        test_date=test_date,
        train_windows=split.train_windows,
        test_windows=split.test_windows,
        metrics=metrics,
        dynamic=dynamic,
        calibration=calibration,
        model=model,
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")

    summary = {
        "symbol": symbol,
        "train_start": train_start.isoformat(),
        "test_date": test_date.isoformat(),
        "train_windows": split.train_windows,
        "test_windows": split.test_windows,
        "metrics": [asdict(row) for row in metrics],
        "dynamic_thresholds": dynamic.replace({np.nan: None}).to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the simple BTC five-minute candle probability table."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--train-start", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--test-date", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/binance"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/2025-12-31"))
    parser.add_argument("--bin-width-bps", type=float, default=1.0)
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--prior-strength", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_backtest(
        symbol=args.symbol.upper(),
        train_start=args.train_start,
        test_date=args.test_date,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        bin_width_bps=args.bin_width_bps,
        min_samples=args.min_samples,
        prior_strength=args.prior_strength,
    )
    all_metrics = next(row for row in summary["metrics"] if row["elapsed_seconds"] == "all")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"\nOverall V1 accuracy: {all_metrics['model_accuracy']:.2%}; "
        f"current-color baseline: {all_metrics['current_color_accuracy']:.2%}"
    )


if __name__ == "__main__":
    main()
