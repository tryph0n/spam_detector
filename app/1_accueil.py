"""Accueil page -- project context and dataset overview."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from text_cleaning import DATA_RAW, ROOT, load_raw_corpus, modelled_corpus_stats

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
def load_split_sizes() -> dict[str, int] | None:
    """Return train/val/test row counts, or None if the splits are not on disk."""
    sizes = {}
    for name in ("train", "val", "test"):
        path = DATA_PROCESSED_DIR / f"{name}.csv"
        if not path.exists():
            return None
        sizes[name] = len(pd.read_csv(path))
    return sizes


@st.cache_data
def load_available_models() -> list[str]:
    """Return names of models whose result JSONs are present on disk."""
    return [
        name for name, fname in MODEL_FILES.items() if (MODELS_DIR / fname).exists()
    ]


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
    stats = modelled_corpus_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SMS delivered (raw file)", f"{stats['total_raw']:,}")
    col2.metric(
        "Ham (legitimate)",
        f"{stats['ham']:,} ({stats['ham'] / stats['total_deduped']:.1%})",
    )
    col3.metric(
        "Spam",
        f"{stats['spam']:,} ({stats['spam'] / stats['total_deduped']:.1%})",
    )
    col4.metric("Duplicates removed", f"{stats['duplicates']:,}")

    st.caption(
        f"The ham and spam counts describe the modelled corpus: "
        f"{stats['total_deduped']:,} messages left after cleaning then deduplicating "
        f"the {stats['total_raw']:,} delivered rows "
        f"({stats['duplicates']:,} removed, "
        f"{stats['duplicates'] / stats['total_raw']:.2%}). Cleaning runs first, so "
        "messages that differ only by case, spacing, a URL or a phone number "
        "collapse into a single row. Every model was trained on this corpus."
    )

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
        title=(
            "Class distribution -- cleaned and deduplicated corpus "
            f"({stats['total_deduped']:,} messages)"
        ),
        xaxis_title="Class",
        yaxis_title="Count",
        showlegend=False,
        height=350,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.warning(f"Raw data not found at `{DATA_RAW.relative_to(ROOT)}`.")

st.markdown("**Source:** SMS Spam Collection dataset (UCI), `data/input/spam.csv`")

splits = load_split_sizes()
if splits:
    st.markdown(
        "**Split strategy:** stratified 70/15/15 with `random_state=42` -- train "
        f"{splits['train']:,} / val {splits['val']:,} / test {splits['test']:,}, "
        f"summing to {sum(splits.values()):,} messages."
    )
else:
    st.markdown(
        "**Split strategy:** stratified 70/15/15 (train/val/test), class ratio preserved."
    )

if DATA_RAW.exists():
    st.subheader("Sample messages")
    st.caption("Raw text as delivered, before cleaning and deduplication.")
    st.dataframe(
        load_raw_corpus().sample(5, random_state=42),
        width="stretch",
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
    else "No trained models found yet -- run `spam-detection_DL.ipynb`."
)

st.markdown(
    f"""
    1. **Text preprocessing** -- lowercase, URL/phone tokenization, then
       deduplication on the cleaned text
    2. **Five deep learning models compared ({trained_count}/{model_count} trained):**
       - **BiLSTM** -- lightweight sequence model
       - **TextCNN** -- convolutional text classifier
       - **RoBERTa** -- state-of-the-art transformer
       - **DistilBERT** -- optimized transformer (distilled from BERT)
       - **BiLSTM GAN-Augmented** -- BiLSTM trained with GAN-synthesized minority
         samples. Its scores are reported on the Resultats page, but its checkpoint
         was saved against a vocabulary rebuilt on the augmented training set that
         was never persisted, so the model cannot be reloaded for inference and the
         Demo page does not serve it.
    3. **Business priority:** High precision on spam class (minimize false positives = avoid blocking legitimate messages)

    {best_model_line}
    """
)
