"""Accueil page -- project context and dataset overview."""

import json
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "input" / "spam.csv"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

MODEL_FILES = {
    "BiLSTM": "bilstm_results.json",
    "TextCNN": "textcnn_results.json",
    "RoBERTa": "roberta_results.json",
    "DistilBERT": "distilbert_results.json",
    "BiLSTM GAN-Augmented": "bilstm_augmented_results.json",
}


@st.cache_data
def load_dataset_stats() -> dict:
    """Compute dataset statistics from source CSV to avoid hardcoding counts."""
    df = pd.read_csv(DATA_RAW, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    total = len(df)
    duplicates = df.duplicated(subset="text").sum()
    df_deduped = df.drop_duplicates(subset="text")
    ham = (df_deduped["label"] == "ham").sum()
    spam = (df_deduped["label"] == "spam").sum()
    return {
        "total_raw": total,
        "duplicates": int(duplicates),
        "ham": int(ham),
        "spam": int(spam),
        "total_deduped": int(ham + spam),
    }


@st.cache_data
def load_available_models() -> list[str]:
    """Return names of models whose result JSONs are present on disk."""
    return [name for name, fname in MODEL_FILES.items() if (MODELS_DIR / fname).exists()]


@st.cache_data
def load_best_model_f1() -> tuple[str, float] | tuple[None, None]:
    """Return (model_name, f1) for the model with highest spam F1, or (None, None)."""
    best_name, best_f1 = None, -1.0
    for name, fname in MODEL_FILES.items():
        path = MODELS_DIR / fname
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if data.get("f1", -1.0) > best_f1:
                best_f1 = data["f1"]
                best_name = name
    return (best_name, best_f1) if best_name else (None, None)


st.title("AT&T Spam Detector")
st.markdown("### Automatic SMS spam detection using deep learning")

st.divider()

st.header("Context")
st.markdown(
    """
    AT&T Inc. is the world's largest telecommunications company by revenue.
    One of the main pain points AT&T users face is constant exposure to **spam messages**.

    AT&T has been manually flagging spam messages, but they need an **automated solution**
    to detect spam based solely on SMS content.
    """
)

st.header("Dataset")

if DATA_RAW.exists():
    stats = load_dataset_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total SMS (raw)", f"{stats['total_raw']:,}")
    col2.metric(
        "Ham (legitimate)",
        f"{stats['ham']:,} ({stats['ham'] / stats['total_deduped']:.0%})",
    )
    col3.metric(
        "Spam",
        f"{stats['spam']:,} ({stats['spam'] / stats['total_deduped']:.0%})",
    )
    col4.metric("Duplicates removed", f"{stats['duplicates']:,}")

    fig = go.Figure(
        data=[
            go.Bar(
                x=["Ham", "Spam"],
                y=[stats["ham"], stats["spam"]],
                marker_color=["#636EFA", "#EF553B"],
                text=[stats["ham"], stats["spam"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Class distribution (after deduplication)",
        xaxis_title="Class",
        yaxis_title="Count",
        showlegend=False,
        height=350,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"Raw data not found at `{DATA_RAW}`.")

st.markdown(
    """
    **Source:** SMS Spam Collection dataset

    **Split strategy:** Stratified 70/15/15 (train/val/test) preserving class ratio.
    """
)

if DATA_RAW.exists():
    df_raw = pd.read_csv(DATA_RAW, encoding="latin-1", usecols=[0, 1])
    df_raw.columns = ["label", "text"]
    st.subheader("Sample messages")
    st.dataframe(
        df_raw.sample(5, random_state=42),
        width='stretch',
        hide_index=True,
    )

st.header("Approach")

available_models = load_available_models()
model_count = len(MODEL_FILES)
trained_count = len(available_models)

best_name, best_f1 = load_best_model_f1()
best_model_line = (
    f"Best model so far: **{best_name}** (spam F1 = {best_f1:.4f})"
    if best_name
    else "No trained models found yet — run the training notebooks."
)

st.markdown(
    f"""
    1. **Text preprocessing** -- lowercase, URL/phone tokenization, deduplication
    2. **Five deep learning models compared ({trained_count}/{model_count} trained):**
       - **BiLSTM** -- lightweight sequence model
       - **TextCNN** -- convolutional text classifier
       - **RoBERTa** -- state-of-the-art transformer
       - **DistilBERT** -- optimized transformer (distilled from BERT)
       - **BiLSTM GAN-Augmented** -- BiLSTM trained with GAN-synthesized minority samples
    3. **Business priority:** High precision on spam class (minimize false positives = avoid blocking legitimate messages)

    {best_model_line}
    """
)
