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

from streamlit_app.data_pipeline import build_data_bundle
from streamlit_app.model_manager import ModelManager

st.set_page_config(
    page_title="Solana 4H Forecast Monitor",
    page_icon="📈",
    layout="wide",
)

import pandas as pd
import plotly.graph_objects as go

st.title("Solana 4-Hour Prediction Dashboard")
st.caption("Live price feed with model forecasts.")

@st.cache_resource(show_spinner=False)
def get_model_manager() -> ModelManager:
    return ModelManager()

@st.cache_data(ttl=60, show_spinner=False)
def load_data():
    return build_data_bundle()

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
