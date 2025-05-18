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

st.title("Solana 4-Hour Prediction Dashboard")
st.caption("Live price feed with model forecasts.")

manager = ModelManager()
bundle = build_data_bundle()

with st.spinner("Loading models..."):
    manager.ensure_models(bundle.feature_frame)

results = manager.predict_all(bundle.feature_frame)

if results:
    st.success(f"Loaded {len(results)} models successfully")
    for name, result in results.items():
        st.subheader(name)
        st.metric("MAE", f"{result.metrics['mae']:.4f}")
