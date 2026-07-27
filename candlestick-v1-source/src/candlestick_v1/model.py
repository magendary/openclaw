from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Prediction:
    p_up: float
    sample_count: int
    radius_bins: int


class ProbabilityTableModel:
    """Two-dimensional empirical P(final Up | time, distance from open)."""

    def __init__(
        self,
        *,
        bin_width_bps: float = 1.0,
        min_samples: int = 200,
        prior_strength: float = 20.0,
        prior_mean: float = 0.5,
        max_radius_bins: int = 30,
    ) -> None:
        if bin_width_bps <= 0:
            raise ValueError("bin_width_bps must be positive")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if prior_strength < 0:
            raise ValueError("prior_strength cannot be negative")
        if not 0 <= prior_mean <= 1:
            raise ValueError("prior_mean must be in [0, 1]")
        self.bin_width_bps = float(bin_width_bps)
        self.min_samples = int(min_samples)
        self.prior_strength = float(prior_strength)
        self.prior_mean = float(prior_mean)
        self.max_radius_bins = int(max_radius_bins)
        self._table: pd.DataFrame | None = None
        self._by_elapsed: dict[int, pd.DataFrame] = {}
        self._elapsed_base_rate: dict[int, float] = {}

    def _to_bin(self, values: pd.Series | np.ndarray | float) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return np.floor(array / self.bin_width_bps).astype(int)

    def fit(self, frame: pd.DataFrame) -> "ProbabilityTableModel":
        required = {"elapsed_seconds", "distance_from_open_bps", "final_up"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Cannot fit on an empty frame")

        work = frame[list(required)].copy()
        work["distance_bin"] = self._to_bin(work["distance_from_open_bps"])
        grouped = (
            work.groupby(["elapsed_seconds", "distance_bin"], as_index=False)
            .agg(up_count=("final_up", "sum"), sample_count=("final_up", "size"))
            .sort_values(["elapsed_seconds", "distance_bin"])
        )
        grouped["p_up_smoothed"] = (
            grouped["up_count"] + self.prior_strength * self.prior_mean
        ) / (grouped["sample_count"] + self.prior_strength)
        self._table = grouped
        self._by_elapsed = {
            int(elapsed): group.set_index("distance_bin").sort_index()
            for elapsed, group in grouped.groupby("elapsed_seconds", sort=True)
        }
        self._elapsed_base_rate = {
            int(elapsed): float(group["final_up"].mean())
            for elapsed, group in work.groupby("elapsed_seconds", sort=True)
        }
        return self

    def _require_fitted(self) -> None:
        if self._table is None:
            raise RuntimeError("Model is not fitted")

    def predict_one(self, elapsed_seconds: int, distance_from_open_bps: float) -> Prediction:
        self._require_fitted()
        elapsed = int(elapsed_seconds)
        if elapsed not in self._by_elapsed:
            raise ValueError(f"Unknown elapsed_seconds={elapsed}")

        group = self._by_elapsed[elapsed]
        center = int(self._to_bin(distance_from_open_bps).item())
        selected: pd.DataFrame | None = None
        selected_radius = 0

        for radius in range(self.max_radius_bins + 1):
            mask = (group.index >= center - radius) & (group.index <= center + radius)
            candidate = group.loc[mask]
            count = int(candidate["sample_count"].sum()) if not candidate.empty else 0
            if count >= self.min_samples:
                selected = candidate
                selected_radius = radius
                break

        if selected is None or selected.empty:
            base = self._elapsed_base_rate[elapsed]
            return Prediction(float(base), 0, self.max_radius_bins + 1)

        up_count = float(selected["up_count"].sum())
        sample_count = int(selected["sample_count"].sum())
        probability = (
            up_count + self.prior_strength * self.prior_mean
        ) / (sample_count + self.prior_strength)
        return Prediction(float(probability), sample_count, selected_radius)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"elapsed_seconds", "distance_from_open_bps"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        predictions = [
            self.predict_one(int(row.elapsed_seconds), float(row.distance_from_open_bps))
            for row in frame.itertuples(index=False)
        ]
        result = frame.copy()
        result["p_up"] = [prediction.p_up for prediction in predictions]
        result["model_sample_count"] = [prediction.sample_count for prediction in predictions]
        result["model_radius_bins"] = [prediction.radius_bins for prediction in predictions]
        result["predicted_up"] = (result["p_up"] >= 0.5).astype(int)
        result["confidence"] = np.maximum(result["p_up"], 1.0 - result["p_up"])
        return result

    def probability_table(self) -> pd.DataFrame:
        self._require_fitted()
        assert self._table is not None
        result = self._table.copy()
        result["distance_bin_lower_bps"] = result["distance_bin"] * self.bin_width_bps
        result["distance_bin_upper_bps"] = (
            result["distance_bin"] + 1
        ) * self.bin_width_bps
        return result[
            [
                "elapsed_seconds",
                "distance_bin_lower_bps",
                "distance_bin_upper_bps",
                "up_count",
                "sample_count",
                "p_up_smoothed",
            ]
        ]
