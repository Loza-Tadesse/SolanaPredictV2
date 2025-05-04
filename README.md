# Solana Prediction Dashboard

This workspace now includes a Streamlit application that visualizes the latest four hours of 1-minute Solana price data and overlays predictions from three models:

- SGDRegressor (scikit-learn) with feature scaling
- Passive-Aggressive Regressor (scikit-learn)
- River online linear regression (incremental learning)

Below the price chart the dashboard shows live metrics that refresh every minute and provides buttons to trigger a full retraining pipeline for each model.

## Running the app

1. Create or activate a Python 3.10+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** PyTorch wheels differ by platform. The provided requirement pins a suitable version for Apple Silicon; adjust as needed for other setups.

3. Launch the Streamlit app from the repository root:

   ```bash
   streamlit run streamlit_app/app.py
   ```

## Architecture overview

- `streamlit_app/data_pipeline.py` handles data ingestion from CryptoCompare, alignment, and feature engineering that mirrors the exploratory notebooks.
- `streamlit_app/pipelines/` houses the SGDRegressor, Passive-Aggressive, and River linear pipelines plus shared utilities.
- `streamlit_app/model_pipelines.py` remains as a compatibility layer that re-exports the active pipelines.
- `streamlit_app/model_manager.py` orchestrates the pipelines, tracks metrics, and computes live evaluation scores.
- `streamlit_app/app.py` is the Streamlit UI that ties everything together.

Models are persisted under `artifacts/models/<model_name>/` and refreshed when retraining is triggered from the UI.
