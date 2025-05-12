"""Utility wrapper that coordinates multiple model pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from streamlit_app.pipelines import (
    BaseModelPipeline,
    PassiveAggressivePipeline,
    RiverLinearPipeline,
    SGDRegressorPipeline,
    default_metrics,
)


@dataclass
class PredictionResult:
    predictions: pd.Series
    metrics: Dict[str, float]


class ModelManager:
    def __init__(self) -> None:
        self.pipelines: Dict[str, BaseModelPipeline] = {}
        self.init_errors: Dict[str, str] = {}
        self._register_default_pipelines()

    def _register_default_pipelines(self) -> None:
        self.pipelines["SGD Regressor"] = SGDRegressorPipeline()
        self.pipelines["Passive-Aggressive"] = PassiveAggressivePipeline()
        try:
            self.pipelines["River Linear"] = RiverLinearPipeline()
        except ImportError as exc:
            self.init_errors["River Linear"] = str(exc)

    def ensure_models(self, feature_frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        for name, pipeline in self.pipelines.items():
            if not pipeline.model_exists():
                metrics[name] = pipeline.train_and_deploy(feature_frame)
        return metrics

    def retrain(self, name: str, feature_frame: pd.DataFrame) -> Dict[str, float]:
        pipeline = self.pipelines.get(name)
        if pipeline is None:
            raise KeyError(f"Model '{name}' is not registered. Available: {list(self.pipelines)}")
        return pipeline.train_and_deploy(feature_frame)

    def update_with_new_data(
        self, full_feature_frame: pd.DataFrame, new_rows: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        for name, pipeline in self.pipelines.items():
            if not pipeline.model_exists():
                metrics[name] = pipeline.train_and_deploy(full_feature_frame)
            elif pipeline.supports_incremental_updates() and not new_rows.empty:
                metrics[name] = pipeline.incremental_update(full_feature_frame, new_rows)
            else:
                metrics[name] = pipeline.train_and_deploy(full_feature_frame)
        return metrics

    def predict_all(
        self,
        feature_frame: pd.DataFrame,
        inference_frame: Optional[pd.DataFrame] = None,
    ) -> Dict[str, PredictionResult]:
        actual = feature_frame["target_price_1min"].copy()
        actual.index = feature_frame.index + pd.Timedelta(minutes=1)
        inference_data = inference_frame if inference_frame is not None else feature_frame
        outcomes: Dict[str, PredictionResult] = {}
        for name, pipeline in self.pipelines.items():
            if not pipeline.model_exists():
                continue
            predictions = pipeline.predict(inference_data)
            metrics = self._compute_live_metrics(actual, predictions)
            outcomes[name] = PredictionResult(predictions=predictions, metrics=metrics)
        return outcomes

    @staticmethod
    def _compute_live_metrics(actual: pd.Series, predictions: pd.Series) -> Dict[str, float]:
        joined = pd.DataFrame({"actual": actual, "pred": predictions}).dropna()
        if joined.empty:
            return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "mape": float("nan")}
        baseline_metrics = default_metrics(joined["actual"].values, joined["pred"].values)
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.abs((joined["pred"] - joined["actual"]) / joined["actual"])
            mape = float(np.nanmean(mape) * 100)
        baseline_metrics["mape"] = mape
        return baseline_metrics

    def get_training_metrics(self) -> Dict[str, Optional[Dict[str, float]]]:
        stats: Dict[str, Optional[Dict[str, float]]] = {}
        for name, pipeline in self.pipelines.items():
            stats[name] = pipeline.load_metrics()
        return stats
