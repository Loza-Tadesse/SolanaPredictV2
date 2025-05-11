"""River linear regression pipeline for minute-ahead Solana deltas."""
from __future__ import annotations

from typing import Dict, Iterable, cast

import joblib
import numpy as np
import pandas as pd
import json

try:  # Optional dependency for online learning.
    from river import compose, linear_model, optim, preprocessing  # type: ignore
except ImportError:  # pragma: no cover - gracefully handled at runtime.
    compose = None  # type: ignore
    linear_model = None  # type: ignore
    optim = None  # type: ignore
    preprocessing = None  # type: ignore

from streamlit_app.data_pipeline import CLEAN_FEATURES

from .base import BaseModelPipeline, TrainResult, default_metrics


def _to_feature_dict(row: Iterable[float]) -> Dict[str, float]:
    return {feature: float(value) for feature, value in zip(CLEAN_FEATURES, row)}


class RiverLinearPipeline(BaseModelPipeline):
    def __init__(self) -> None:
        super().__init__(name="river_linear")
        if linear_model is None or optim is None or preprocessing is None or compose is None:
            raise ImportError(
                "The 'river' package is required for the RiverLinearPipeline. Install it via 'pip install river'."
            )
        self.model_path = self.artifact_dir / "model.joblib"
        self.metadata_path = self.artifact_dir / "metadata.json"
        self.model_version = "river-linear-lr0.001-epochs6"
        self.learning_rate = 0.001
        self.training_epochs = 6

    def model_exists(self) -> bool:
        if not self.model_path.exists() or not self.metadata_path.exists():
            return False
        try:
            with self.metadata_path.open("r", encoding="utf-8") as fp:
                metadata = json.load(fp)
        except (json.JSONDecodeError, OSError):
            return False
        return metadata.get("model_version") == self.model_version

    def _load_model_from_disk(self) -> object:
        return joblib.load(self.model_path)

    def _persist_model(self, result: TrainResult) -> None:
        joblib.dump(result.model, self.model_path)
        metadata = {
            "model_version": self.model_version,
            "learning_rate": self.learning_rate,
            "training_epochs": self.training_epochs,
        }
        with self.metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

    def _build_model(self):  # type: ignore[override]
        optimizer = optim.SGD(lr=self.learning_rate)
        regressor = linear_model.LinearRegression(optimizer=optimizer, intercept_init=0.0)
        return compose.Pipeline(preprocessing.StandardScaler(), regressor)

    def _train_model(self, feature_frame: pd.DataFrame) -> TrainResult:
        features = feature_frame[CLEAN_FEATURES]
        target_prices = feature_frame["target_price_1min"].values
        base_prices = feature_frame["price"].values
        target_deltas = target_prices - base_prices

        split_index = int(len(features) * 0.8)
        train_features = features.iloc[:split_index]
        train_targets = target_deltas[:split_index]
        valid_features = features.iloc[split_index:]
        valid_targets = target_prices[split_index:]
        valid_base = base_prices[split_index:]

        model = self._build_model()
        train_pairs = [
            (_to_feature_dict(row_values), float(target))
            for row_values, target in zip(train_features.values, train_targets, strict=False)
        ]
        for _ in range(self.training_epochs):
            for features_dict, target in train_pairs:
                model.learn_one(features_dict, target)

        if len(valid_features) == 0:
            metrics = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
        else:
            delta_preds = [
                model.predict_one(_to_feature_dict(row_values))
                for row_values in valid_features.values
            ]
            pred_prices = valid_base + np.array(delta_preds)
            metrics = default_metrics(valid_targets, pred_prices)

        return TrainResult(model=model, metrics=metrics)

    def _predict_with_model(self, model: object, feature_frame: pd.DataFrame) -> pd.Series:
        river_model = cast(object, model)
        feature_matrix = feature_frame[CLEAN_FEATURES].values
        base_prices = feature_frame["price"].values

        delta_predictions = [
            river_model.predict_one(_to_feature_dict(row_values))
            for row_values in feature_matrix
        ]
        predicted_prices = base_prices + np.array(delta_predictions)
        index = feature_frame.index + pd.Timedelta(minutes=1)
        return pd.Series(predicted_prices, index=index, name=self.name)

    def supports_incremental_updates(self) -> bool:
        return True

    def incremental_update(
        self, full_feature_frame: pd.DataFrame, new_rows: pd.DataFrame
    ) -> Dict[str, float]:
        if new_rows.empty:
            return {}
        river_model = cast(object, self.ensure_model())

        features = new_rows[CLEAN_FEATURES].values
        target_prices = new_rows["target_price_1min"].values
        base_prices = new_rows["price"].values
        deltas = target_prices - base_prices

        incremental_pairs = [
            (_to_feature_dict(row_values), float(target_delta))
            for row_values, target_delta in zip(features, deltas, strict=False)
        ]
        for features_dict, target_delta in incremental_pairs:
            river_model.learn_one(features_dict, target_delta)

        joblib.dump(river_model, self.model_path)
        self._model_cache = river_model

        preds = [
            river_model.predict_one(_to_feature_dict(row_values))
            for row_values in features
        ]
        predicted_prices = base_prices + np.array(preds)
        metrics = default_metrics(target_prices, predicted_prices)
        self.save_metrics(metrics)
        return metrics
