"""Model pipeline implementations."""
from .base import BaseModelPipeline, TrainResult, default_metrics
from .passive_aggressive import PassiveAggressivePipeline
from .river_linear import RiverLinearPipeline
from .sgd_regressor import SGDRegressorPipeline

__all__ = [
    "BaseModelPipeline",
    "TrainResult",
    "default_metrics",
    "PassiveAggressivePipeline",
    "RiverLinearPipeline",
    "SGDRegressorPipeline",
]
