"""Streamlit dashboard for monitoring Solana price forecasts."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

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

import pandas as pd
import plotly.graph_objects as go

st.title("Solana 4-Hour Prediction Dashboard")
st.caption("Live price feed with model forecasts.")

@st.cache_resource(show_spinner=False)
def get_model_manager() -> ModelManager:
    return ModelManager()

@st.cache_data(ttl=60, show_spinner=False)
def load_data(minutes: int = DEFAULT_MINUTES):
    return build_data_bundle(minutes=minutes)

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

with st.spinner("Loading models..."):
    manager.ensure_models(bundle.feature_frame)

results = manager.predict_all(bundle.feature_frame)
actual_price = bundle.aligned_price_frame["price"].iloc[-30:]

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

# Metrics
for name, result in results.items():
    st.subheader(name)
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{result.metrics['mae']:.4f}")
    col2.metric("RMSE", f"{result.metrics['rmse']:.4f}")
    col3.metric("R²", f"{result.metrics.get('r2', 0):.4f}")
