"""Train and persist baseline models so the Streamlit app starts warm."""
from __future__ import annotations

import sys
from pathlib import Path

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from streamlit_app.data_pipeline import build_data_bundle
    from streamlit_app.model_manager import ModelManager

    bundle = build_data_bundle()
    manager = ModelManager()
    metrics = manager.ensure_models(bundle.feature_frame)
    if not metrics:
        print("Models already trained; nothing to do.")
    else:
        for name, stats in metrics.items():
            print(f"{name} trained. Metrics: {stats}")


if __name__ == "__main__":
    main()
