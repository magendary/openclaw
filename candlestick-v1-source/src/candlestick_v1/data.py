from __future__ import annotations

import io
import math
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BINANCE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    train_windows: int
    test_windows: int


def iter_months(start: date, end: date) -> Iterable[tuple[int, int]]:
    """Yield calendar months intersecting the inclusive date range."""
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        month += 1
        if month == 13:
            year += 1
            month = 1


def _timestamp_unit(values: pd.Series) -> str:
    """Binance Spot timestamps are ms before 2025 and us from 2025 onward."""
    sample = int(values.dropna().iloc[0])
    if sample >= 10**15:
        return "us"
    if sample >= 10**12:
        return "ms"
    raise ValueError(f"Unexpected Binance timestamp magnitude: {sample}")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "candlestick-v1/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"Downloaded empty file: {url}")
    destination.write_bytes(payload)


def load_monthly_klines(
    *,
    symbol: str,
    interval: str,
    year: int,
    month: int,
    cache_dir: Path,
) -> pd.DataFrame:
    """Download and parse one Binance Vision monthly kline archive."""
    filename = f"{symbol}-{interval}-{year:04d}-{month:02d}.zip"
    url = (
        "https://data.binance.vision/data/spot/monthly/klines/"
        f"{symbol}/{interval}/{filename}"
    )
    archive_path = cache_dir / filename
    if not archive_path.exists():
        _download(url, archive_path)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError(
                    f"Expected exactly one CSV in {archive_path}, found {csv_names}"
                )
            with archive.open(csv_names[0]) as handle:
                raw = handle.read()
    except zipfile.BadZipFile as exc:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(f"Corrupt archive: {archive_path}") from exc

    frame = pd.read_csv(
        io.BytesIO(raw),
        header=None,
        names=BINANCE_COLUMNS,
        usecols=["open_time", "open", "high", "low", "close"],
        dtype={"open_time": "int64", "open": "float64", "high": "float64", "low": "float64", "close": "float64"},
    )
    unit = _timestamp_unit(frame["open_time"])
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit=unit, utc=True)
    return frame.sort_values("open_time").reset_index(drop=True)


def load_klines(
    *,
    symbol: str,
    interval: str,
    start_date: date,
    end_date: date,
    cache_dir: Path,
) -> pd.DataFrame:
    frames = [
        load_monthly_klines(
            symbol=symbol,
            interval=interval,
            year=year,
            month=month,
            cache_dir=cache_dir,
        )
        for year, month in iter_months(start_date, end_date)
    ]
    if not frames:
        raise ValueError("No monthly data requested")

    data = pd.concat(frames, ignore_index=True)
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    data = data[(data["open_time"] >= start_ts) & (data["open_time"] < end_ts)]
    data = data.drop_duplicates(subset=["open_time"], keep="last")
    data = data.sort_values("open_time").reset_index(drop=True)
    if data.empty:
        raise ValueError("No kline rows remain after date filtering")
    return data


def build_five_minute_snapshots(one_minute: pd.DataFrame) -> pd.DataFrame:
    """Create 60/120/180/240-second snapshots for complete five-minute windows."""
    required = {"open_time", "open", "close"}
    missing = required - set(one_minute.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame = one_minute.copy()
    frame["window_start"] = frame["open_time"].dt.floor("5min")
    frame["minute_index"] = (
        (frame["open_time"] - frame["window_start"]).dt.total_seconds() // 60
    ).astype(int)

    records: list[dict[str, object]] = []
    for window_start, group in frame.groupby("window_start", sort=True):
        group = group.sort_values("open_time")
        if len(group) != 5 or group["minute_index"].tolist() != [0, 1, 2, 3, 4]:
            continue

        open_price = float(group.iloc[0]["open"])
        final_close = float(group.iloc[-1]["close"])
        final_up = int(final_close >= open_price)

        for minute_index in range(4):
            current_price = float(group.iloc[minute_index]["close"])
            elapsed_seconds = (minute_index + 1) * 60
            distance_bps = 10_000.0 * math.log(current_price / open_price)
            records.append(
                {
                    "window_start": window_start,
                    "elapsed_seconds": elapsed_seconds,
                    "open_price": open_price,
                    "current_price": current_price,
                    "distance_from_open_bps": distance_bps,
                    "final_close": final_close,
                    "final_up": final_up,
                }
            )

    snapshots = pd.DataFrame.from_records(records)
    if snapshots.empty:
        raise ValueError("No complete five-minute windows were produced")
    snapshots["window_start"] = pd.to_datetime(snapshots["window_start"], utc=True)
    return snapshots.sort_values(["window_start", "elapsed_seconds"]).reset_index(drop=True)


def split_snapshots(snapshots: pd.DataFrame, test_date: date) -> DatasetSplit:
    test_start = pd.Timestamp(test_date, tz="UTC")
    test_end = test_start + pd.Timedelta(days=1)
    train = snapshots[snapshots["window_start"] < test_start].copy()
    test = snapshots[
        (snapshots["window_start"] >= test_start)
        & (snapshots["window_start"] < test_end)
    ].copy()

    if train.empty:
        raise ValueError("Training split is empty")
    if test.empty:
        raise ValueError("Test split is empty")

    train_windows = int(train["window_start"].nunique())
    test_windows = int(test["window_start"].nunique())
    return DatasetSplit(train, test, train_windows, test_windows)
