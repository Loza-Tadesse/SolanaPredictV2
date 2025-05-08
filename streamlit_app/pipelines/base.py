"""Shared utilities for model pipelines."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "models"
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)


def default_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R² for the provided arrays."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r2 = r2_score(y_true, y_pred)
    except ValueError:
        r2 = float("nan")
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


@dataclass
class TrainResult:
    model: object
    metrics: Dict[str, float]
    extra: Optional[Dict[str, object]] = None


class BaseModelPipeline(ABC):
    """Common utilities for individual model pipelines."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        self.artifact_dir = ARTIFACTS_ROOT / name
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.artifact_dir / "metrics.json"
        self._model_cache: Optional[object] = None

    @abstractmethod
    def model_exists(self) -> bool:
        """Return True when the persisted model is ready for inference."""

    @abstractmethod
    def _load_model_from_disk(self) -> object:
        """Load the persisted model artifacts from disk."""

    @abstractmethod
    def _persist_model(self, result: TrainResult) -> None:
        """Persist the trained model (and extra assets) to disk."""

    @abstractmethod
    def _train_model(self, feature_frame: pd.DataFrame) -> TrainResult:
        """Fit the model and return the trained instance with metrics."""

    @abstractmethod
    def _predict_with_model(self, model: object, feature_frame: pd.DataFrame) -> pd.Series:
        """Generate predictions given the loaded model."""

    def ensure_model(self) -> object:
        if self._model_cache is None:
            if not self.model_exists():
                raise RuntimeError(
                    f"Model '{self.name}' is not available. Trigger a retrain first."
                )
            self._model_cache = self._load_model_from_disk()
        return self._model_cache

    def train_and_deploy(self, feature_frame: pd.DataFrame) -> Dict[str, float]:
        result = self._train_model(feature_frame)
        self._persist_model(result)
        self.save_metrics(result.metrics)
        self._model_cache = result.model
        return result.metrics

    def predict(self, feature_frame: pd.DataFrame) -> pd.Series:
        model = self.ensure_model()
        return self._predict_with_model(model, feature_frame)

    def save_metrics(self, metrics: Dict[str, float]) -> None:
        with self.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)

    def load_metrics(self) -> Optional[Dict[str, float]]:
        if not self.metrics_path.exists():
            return None
        with self.metrics_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def supports_incremental_updates(self) -> bool:
        return False

    def incremental_update(
        self, full_feature_frame: pd.DataFrame, new_rows: pd.DataFrame
    ) -> Dict[str, float]:
        raise NotImplementedError(
            f"Pipeline '{self.name}' does not implement incremental updates."
        )
