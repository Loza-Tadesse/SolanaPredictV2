"""PassiveAggressiveRegressor pipeline for minute-ahead Solana deltas."""
from __future__ import annotations

from typing import Dict, cast

import joblib
import pandas as pd
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from streamlit_app.data_pipeline import CLEAN_FEATURES

from .base import BaseModelPipeline, TrainResult, default_metrics


class PassiveAggressivePipeline(BaseModelPipeline):
    def __init__(self) -> None:
        super().__init__(name="passive_aggressive")
        self.model_path = self.artifact_dir / "model.joblib"

    def model_exists(self) -> bool:
        return self.model_path.exists()

    def _load_model_from_disk(self) -> object:
        return joblib.load(self.model_path)

    def _persist_model(self, result: TrainResult) -> None:
        joblib.dump(result.model, self.model_path)

    def _train_model(self, feature_frame: pd.DataFrame) -> TrainResult:
        features = feature_frame[CLEAN_FEATURES].values
        target_prices = feature_frame["target_price_1min"].values
        base_prices = feature_frame["price"].values
        target_deltas = target_prices - base_prices

        split_index = int(len(features) * 0.8)
        x_train, x_valid = features[:split_index], features[split_index:]
        y_train = target_deltas[:split_index]

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "pa",
                    PassiveAggressiveRegressor(
                        C=0.5,
                        epsilon=1e-3,
                        loss="squared_epsilon_insensitive",
                        max_iter=2000,
                        tol=1e-4,
                        random_state=42,
                        shuffle=True,
                    ),
                ),
            ],
            memory=None,
        )
        pipeline.fit(x_train, y_train)

        if len(x_valid) == 0:
            metrics = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
        else:
            delta_pred = pipeline.predict(x_valid)
            pred_prices = base_prices[split_index:] + delta_pred
            metrics = default_metrics(target_prices[split_index:], pred_prices)

        return TrainResult(model=pipeline, metrics=metrics)

    def _predict_with_model(self, model: object, feature_frame: pd.DataFrame) -> pd.Series:
        pipeline = cast(Pipeline, model)
        delta_preds = pipeline.predict(feature_frame[CLEAN_FEATURES].values)
        prices = feature_frame["price"].values + delta_preds
        index = feature_frame.index + pd.Timedelta(minutes=1)
        return pd.Series(prices, index=index, name=self.name)

    def supports_incremental_updates(self) -> bool:
        return True

    def incremental_update(
        self, full_feature_frame: pd.DataFrame, new_rows: pd.DataFrame
    ) -> Dict[str, float]:
        if new_rows.empty:
            return {}
        pipeline = cast(Pipeline, self.ensure_model())
        scaler = cast(StandardScaler, pipeline.named_steps["scaler"])
        regressor = cast(PassiveAggressiveRegressor, pipeline.named_steps["pa"])

        feature_values = new_rows[CLEAN_FEATURES].values
        target_prices = new_rows["target_price_1min"].values
        base_prices = new_rows["price"].values
        deltas = target_prices - base_prices

        scaler.partial_fit(feature_values)
        transformed = scaler.transform(feature_values)
        regressor.partial_fit(transformed, deltas)

        pipeline.named_steps["scaler"] = scaler
        pipeline.named_steps["pa"] = regressor

        joblib.dump(pipeline, self.model_path)
        self._model_cache = pipeline

        preds = regressor.predict(transformed) + base_prices
        metrics = default_metrics(target_prices, preds)
        self.save_metrics(metrics)
        return metrics
