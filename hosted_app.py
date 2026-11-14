"""Streamlit Community Cloud entry point for the public synthetic demo."""

import runpy
from pathlib import Path
import streamlit as st
from scripts.fetch_model import main as fetch_weights

ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner="Подготавливаем локальную языковую модель…")
def prepare_weights():
    import os
    from filelock import FileLock

    if (
        os.environ.get("NAV_LLM_MODEL")
        and Path(os.environ["NAV_LLM_MODEL"]).is_file()
        and (
            Path(os.getenv("NAV_RERANK_DIR", ROOT / "models/reranker"))
            / "reranker.onnx"
        ).is_file()
    ):
        return True
    with FileLock(str(ROOT / "models" / "setup.lock"), timeout=600):
        fetch_weights(ROOT / "models")
    return True


prepare_weights()
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
