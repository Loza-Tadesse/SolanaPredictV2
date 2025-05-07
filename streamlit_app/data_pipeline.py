"""Data ingestion and feature engineering utilities for the Streamlit app."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests

CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/histominute"
DEFAULT_SYMBOL = "SOL"
DEFAULT_QUOTE = "USD"
DEFAULT_MINUTES = 240

CLEAN_FEATURES = [
    "price",
    "volume",
    "market_cap",
    "sma_7",
    "sma_14",
    "sma_30",
    "ema_7",
    "ema_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "rsi",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "price_change_1h",
    "price_change_24h",
    "price_change_7d",
    "volume_sma",
    "volume_ratio",
    "volatility",
    "high_14d",
    "low_14d",
    "price_position",
]


def fetch_minute_data(
    symbol: str = DEFAULT_SYMBOL,
    quote: str = DEFAULT_QUOTE,
    minutes: int = DEFAULT_MINUTES,
) -> pd.DataFrame:
    """Fetch recent 1-minute candles for the requested symbol/quote pair."""
    params = {
        "fsym": symbol.upper(),
        "tsym": quote.upper(),
        "limit": minutes,
        "aggregate": 1,
    }
    response = requests.get(CRYPTOCOMPARE_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get("Response") != "Success":
        message = payload.get("Message", "Unknown CryptoCompare error")
        raise RuntimeError(f"CryptoCompare API error: {message}")

    frame = pd.DataFrame(payload["Data"]["Data"])
    if frame.empty:
        raise RuntimeError("CryptoCompare minute endpoint returned no data")

    frame["datetime"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame = frame.rename(
        columns={
            "close": "price",
            "volumefrom": "volume",
            "volumeto": "market_cap",
        }
    )
    frame = frame[["datetime", "price", "volume", "market_cap"]]
    frame = frame.set_index("datetime").sort_index()
    return frame


def align_minute_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the frame is aligned to a continuous minute grid."""
    if frame.empty:
        return frame

    cleaned = frame[~frame.index.duplicated(keep="last")]
    # Use the observed span to define the reindexed grid.
    start = cleaned.index.min()
    end = cleaned.index.max()
    minute_index = pd.date_range(start=start, end=end, freq="1min", tz="UTC")
    aligned = cleaned.reindex(minute_index)
    aligned = aligned.interpolate(method="time").bfill().ffill()
    return aligned


def add_technical_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the technical indicator feature set used across the pipelines."""
    df = frame.copy()

    # Rolling windows measured in minutes.
    sma_short = 60
    sma_medium = 360
    sma_long = 1440
    bb_window = 120
    rsi_period = 30
    price_change_1h = 60
    price_change_24h = 1440
    price_change_7d = 10080
    volume_window = 180
    volatility_window = 120
    position_window = 60 * 24 * 14

    df["sma_7"] = df["price"].rolling(window=sma_short, min_periods=1).mean()
    df["sma_14"] = df["price"].rolling(window=sma_medium, min_periods=1).mean()
    df["sma_30"] = df["price"].rolling(window=sma_long, min_periods=1).mean()
    df["ema_7"] = df["price"].ewm(span=sma_short, adjust=False).mean()
    df["ema_14"] = df["price"].ewm(span=sma_medium, adjust=False).mean()

    ema_fast = df["price"].ewm(span=12, adjust=False).mean()
    ema_slow = df["price"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    delta = df["price"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
    rs = gain / loss.replace({0: np.nan})
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)

    df["bb_middle"] = df["price"].rolling(window=bb_window, min_periods=1).mean()
    bb_std = df["price"].rolling(window=bb_window, min_periods=1).std().fillna(0.0)
    df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
    df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

    df["price_change_1h"] = df["price"].pct_change(price_change_1h).fillna(0.0)
    df["price_change_24h"] = df["price"].pct_change(price_change_24h).fillna(0.0)
    df["price_change_7d"] = df["price"].pct_change(price_change_7d).fillna(0.0)

    df["volume_sma"] = df["volume"].rolling(window=volume_window, min_periods=1).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]
    df["volume_ratio"] = df["volume_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    df["volatility"] = df["price"].rolling(window=volatility_window, min_periods=1).std().fillna(0.0)

    df["high_14d"] = df["price"].rolling(window=position_window, min_periods=1).max()
    df["low_14d"] = df["price"].rolling(window=position_window, min_periods=1).min()
    range_span = (df["high_14d"] - df["low_14d"]).replace(0, np.nan)
    df["price_position"] = (df["price"] - df["low_14d"]) / range_span
    df["price_position"] = df["price_position"].clip(0.0, 1.0).fillna(0.5)

    return df


def create_prediction_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Create supervised learning targets for different forecast horizons."""
    df = frame.copy()
    df["target_price_1min"] = df["price"].shift(-1)
    df["target_price_5min"] = df["price"].shift(-5)
    df["target_price_1h"] = df["price"].shift(-60)
    df["target_price_24h"] = df["price"].shift(-1440)

    for col in ["target_price_5min", "target_price_1h", "target_price_24h"]:
        df[col] = df[col].fillna(df["price"])

    df["target_direction_1min"] = (df["target_price_1min"] > df["price"]).astype(int)
    df["target_direction_5min"] = (df["target_price_5min"] > df["price"]).astype(int)
    df["target_direction_1h"] = (df["target_price_1h"] > df["price"]).astype(int)
    df["target_direction_24h"] = (df["target_price_24h"] > df["price"]).astype(int)

    df["target_change_1min"] = (df["target_price_1min"] - df["price"]) / df["price"] * 100
    df["target_change_5min"] = (df["target_price_5min"] - df["price"]) / df["price"] * 100
    df["target_change_1h"] = (df["target_price_1h"] - df["price"]) / df["price"] * 100
    df["target_change_24h"] = (df["target_price_24h"] - df["price"]) / df["price"] * 100

    change_cols = [
        "target_change_1min",
        "target_change_5min",
        "target_change_1h",
        "target_change_24h",
    ]
    df[change_cols] = df[change_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


@dataclass
class DataBundle:
    """Container for the aligned raw data, training features, and inference frame."""

    aligned_price_frame: pd.DataFrame
    feature_frame: pd.DataFrame
    inference_frame: pd.DataFrame

    @property
    def feature_matrix(self) -> np.ndarray:
        return self.feature_frame[CLEAN_FEATURES].values

    @property
    def target_vector(self) -> np.ndarray:
        return self.feature_frame["target_price_1min"].values


def build_data_bundle(minutes: int = DEFAULT_MINUTES) -> DataBundle:
    """Fetch, align, and feature-engineer the latest crypto minute data."""
    raw = fetch_minute_data(minutes=minutes)
    aligned = align_minute_frame(raw)
    enriched = add_technical_indicators(aligned)
    enriched = create_prediction_targets(enriched)
    enriched = enriched.replace([np.inf, -np.inf], np.nan)
    enriched = enriched.dropna(subset=["price", "volume", "market_cap"])

    training_frame = enriched.dropna(subset=["target_price_1min"]).copy()
    inference_frame = enriched.copy()

    # Fill remaining indicator gaps while preserving target availability information.
    training_frame = training_frame.fillna(0.0)
    filled_inference = inference_frame.fillna(0.0)
    filled_inference["target_price_1min"] = inference_frame["target_price_1min"]

    if len(training_frame) > minutes:
        training_frame = training_frame.iloc[-minutes:]
    if len(filled_inference) > minutes + 1:
        filled_inference = filled_inference.iloc[-(minutes + 1):]

    return DataBundle(
        aligned_price_frame=aligned,
        feature_frame=training_frame,
        inference_frame=filled_inference,
    )


def compute_prediction_alignment(feature_frame: pd.DataFrame) -> pd.Index:
    """Return the timestamps that predictions should be aligned to."""
    return feature_frame.index + timedelta(minutes=1)
