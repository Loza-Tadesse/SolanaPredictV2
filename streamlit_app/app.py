"""Streamlit dashboard for monitoring Solana price forecasts."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

CARD_STYLE = """
    <style>
    .model-card {
        position: relative;
        border-radius: 18px;
        padding: 28px 24px;
        margin: 18px 0;
        color: #f8faff;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(236, 72, 153, 0.12));
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 18px 30px rgba(15, 23, 42, 0.18);
        backdrop-filter: blur(18px);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .model-card::before {
        content: "";
        position: absolute;
        inset: -40% -40% auto auto;
        height: 180px;
        width: 180px;
        background: radial-gradient(circle, rgba(96, 165, 250, 0.45) 0%, rgba(59, 130, 246, 0) 65%);
        transform: rotate(25deg);
        z-index: 0;
    }
    .model-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 26px 38px rgba(15, 23, 42, 0.24);
    }
    .card-title {
        position: relative;
        z-index: 1;
        font-size: 1.45em;
        font-weight: 600;
        margin-bottom: 18px;
        letter-spacing: 0.02em;
    }
    .metric-row {
        position: relative;
        z-index: 1;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin: 12px 0;
        padding: 12px 14px;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .metric-label {
        font-weight: 500;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        font-size: 0.72rem;
        color: rgba(226, 232, 240, 0.82);
    }
    .metric-value {
        font-weight: 600;
        font-size: 1.15em;
        color: #fefce8;
    }
    </style>
"""

# Ensure the project root is on sys.path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from streamlit_app.data_pipeline import build_data_bundle, DEFAULT_MINUTES
from streamlit_app.model_manager import ModelManager

st.set_page_config(
    page_title="Solana 4H Forecast Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DISPLAY_MINUTES = 30
AUTO_RETRAIN_STATE_KEY = "auto_retrain_state"

MODEL_MANAGER_VERSION = "2025-05-24-inference-updates"

PROJECT_DESCRIPTION = (
    "Real-time dashboard for monitoring Solana forecasts with incremental learning, minute-by-minute updates, "
    "and performance metrics. It combines an SGD regressor tuned for smooth trend following, a "
    "Passive-Aggressive regressor engineered to react quickly to momentum shifts, and a River online linear model "
    "that adapts continuously with lightweight updates—giving you a layered view of price dynamics."
)


def _slugify_name(name: str) -> str:
    """Create a consistent lowercase identifier for model names."""
    return name.lower().replace(" ", "_").replace("-", "_")


def _format_timestamp(ts: pd.Timestamp) -> str:
    """Format timestamp to UTC display string."""
    return ts.strftime("%H:%M UTC")


def _format_prediction(value: float) -> str:
    """Format prediction value with 2 decimal places."""
    return f"${value:.2f}"


def _format_optional_timestamp(ts: Optional[pd.Timestamp]) -> str:
    """Gracefully format optional timestamps for display."""
    return _format_timestamp(ts) if ts is not None else "--"


def _milliseconds_until_next_minute() -> int:
    """Calculate milliseconds until the next minute boundary."""
    now = datetime.now(timezone.utc)
    next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    return int((next_minute - now).total_seconds() * 1000)


import pandas as pd
import plotly.graph_objects as go

HEADER_STYLE = """
<style>
.header-metrics {
    display: flex;
    justify-content: center;
    gap: 18px;
    margin: 18px 0 28px;
}
.header-metric {
    min-width: 220px;
    padding: 18px 22px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.35), rgba(236, 72, 153, 0.25));
    text-align: center;
}
.header-metric .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #cbd5f5;
}
.header-metric .value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #fdf4ff;
}
</style>
"""

st.markdown(HEADER_STYLE, unsafe_allow_html=True)
st.markdown("<h1 class='page-title'>Solana 4-Hour Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='page-description'>{PROJECT_DESCRIPTION}</p>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_model_manager(version: str = MODEL_MANAGER_VERSION) -> ModelManager:
    _ = version
    return ModelManager()

@st.cache_data(ttl=55, show_spinner=False)
def load_data(minutes: int = DEFAULT_MINUTES):
    return build_data_bundle(minutes=minutes)

def _milliseconds_until_next_minute(now: datetime | None = None) -> int:
    current = now or datetime.now(timezone.utc)
    next_minute = (current.replace(second=0, microsecond=0) + timedelta(minutes=1))
    delta = next_minute - current
    millis = int(delta.total_seconds() * 1000)
    return max(millis, 250)


def display_model_cards(
    metrics_dict: Dict[str, Dict],
    latest_predictions: Dict[str, float],
    prediction_timestamps: Dict[str, Optional[pd.Timestamp]],
    fallback_ts: pd.Timestamp | None,
) -> None:
    """Display model performance cards with enhanced styling."""
    st.markdown(CARD_STYLE, unsafe_allow_html=True)
    
    cols = st.columns(3)
    model_names = ["sgd_regressor", "passive_aggressive", "river_linear"]
    display_titles = ["SGD Regressor", "Passive-Aggressive", "River Linear"]
    
    for idx, (model_name, title) in enumerate(zip(model_names, display_titles)):
        with cols[idx]:
            metrics = metrics_dict.get(model_name, {})
            mae = metrics.get("mae", 0.0)
            rmse = metrics.get("rmse", 0.0)
            prediction = latest_predictions.get(model_name)
            timestamp = prediction_timestamps.get(model_name) or fallback_ts
            forecast_display = _format_timestamp(timestamp) if timestamp is not None else "--"
            
            card_html = f"""
            <div class="model-card">
                <div class="card-title">{title}</div>
                <div class="metric-row">
                    <span class="metric-label">Forecast for:</span>
                    <span class="metric-value">{forecast_display}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Predicted Price:</span>
                    <span class="metric-value">{_format_prediction(prediction) if prediction is not None else "--"}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">MAE:</span>
                    <span class="metric-value">${mae:.4f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">RMSE:</span>
                    <span class="metric-value">${rmse:.4f}</span>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)


interval_ms = _milliseconds_until_next_minute()
_ = st_autorefresh(interval=interval_ms, limit=None, debounce=False, key="solana_price_refresh")

def _auto_retrain_models(manager, feature_frame, latest_timestamp):
    state = st.session_state.setdefault(
        AUTO_RETRAIN_STATE_KEY, {"timestamp": None, "metrics": {}}
    )
    stored_timestamp = state.get("timestamp")
    previous_ts = pd.Timestamp(stored_timestamp) if stored_timestamp else None

    if latest_timestamp is None or (previous_ts and latest_timestamp <= previous_ts):
        return state.get("metrics", {}), False

    if previous_ts is None:
        st.session_state[AUTO_RETRAIN_STATE_KEY] = {
            "timestamp": latest_timestamp.isoformat(),
            "metrics": state.get("metrics", {}),
        }
        return state.get("metrics", {}), False

    new_rows = feature_frame[feature_frame.index > previous_ts]
    if new_rows.empty:
        return state.get("metrics", {}), False

    with st.spinner("Updating models..."):
        metrics_snapshot = manager.update_with_new_data(feature_frame, new_rows)

    st.session_state[AUTO_RETRAIN_STATE_KEY] = {
        "timestamp": latest_timestamp.isoformat(),
        "metrics": metrics_snapshot,
    }
    return metrics_snapshot, True

manager = get_model_manager()
bundle = load_data()

st.markdown(CARD_STYLE, unsafe_allow_html=True)

with st.spinner("Loading models..."):
    manager.ensure_models(bundle.feature_frame)

latest_timestamp = bundle.feature_frame.index[-1] if not bundle.feature_frame.empty else None
metrics_snapshot, retrained = _auto_retrain_models(manager, bundle.feature_frame, latest_timestamp)

# Prepare predictions
if bundle.inference_frame is not None and not bundle.inference_frame.empty:
    results = manager.predict_all(bundle.feature_frame, bundle.inference_frame)
else:
    results = manager.predict_all(bundle.feature_frame)

forecast_ts = None
for result in results.values():
    if not result.predictions.empty:
        forecast_ts = result.predictions.index[-1]
        break

if forecast_ts is None and latest_timestamp is not None:
    forecast_ts = latest_timestamp + pd.Timedelta(minutes=1)

# Display header metrics
current_price = bundle.aligned_price_frame["price"].iloc[-1]
predictions_summary = {
    _slugify_name(name): res.predictions.iloc[-1] for name, res in results.items()
}
prediction_timestamps = {
    _slugify_name(name): (res.predictions.index[-1] if not res.predictions.empty else None)
    for name, res in results.items()
}

sgd_forecast_time = prediction_timestamps.get("sgd_regressor") or forecast_ts
pa_forecast_time = prediction_timestamps.get("passive_aggressive") or forecast_ts
river_forecast_time = prediction_timestamps.get("river_linear") or forecast_ts


header_html = f"""
<div class="header-metrics">
    <div class="header-metric">
        <div class="label">Current Price</div>
        <div class="value">{_format_prediction(current_price)}</div>
    </div>
    <div class="header-metric">
        <div class="label">SGD Forecast ({_format_optional_timestamp(sgd_forecast_time)})</div>
        <div class="value">{_format_prediction(predictions_summary.get('sgd_regressor', 0))}</div>
    </div>
    <div class="header-metric">
        <div class="label">PA Forecast ({_format_optional_timestamp(pa_forecast_time)})</div>
        <div class="value">{_format_prediction(predictions_summary.get('passive_aggressive', 0))}</div>
    </div>
    <div class="header-metric">
        <div class="label">River Forecast ({_format_optional_timestamp(river_forecast_time)})</div>
        <div class="value">{_format_prediction(predictions_summary.get('river_linear', 0))}</div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

actual_price = bundle.aligned_price_frame["price"].iloc[-DISPLAY_MINUTES:]

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=actual_price.index,
    y=actual_price.values,
    name="Actual Price",
    mode="lines"
))

for name, result in results.items():
    preds = result.predictions.iloc[-30:]
    fig.add_trace(go.Scatter(
        x=preds.index,
        y=preds.values,
        name=name,
        mode="lines",
        line={"dash": "dash"}
    ))

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Model Performance")

# Use the new display_model_cards function
metrics_dict = {
    _slugify_name(name): result.metrics for name, result in results.items()
}
display_model_cards(metrics_dict, predictions_summary, prediction_timestamps, forecast_ts)

