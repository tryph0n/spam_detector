"""Corpus loading and text normalization shared by the Streamlit pages.

`clean_text` mirrors cell 13 of `spam-detection_DL.ipynb`; the two must
stay identical, because the pages count a corpus the models were trained on and
the demo feeds the vocabulary that was built from that same normalization.

Two corpus referentials exist in this project and must never be confused:
`load_raw_corpus` returns the file exactly as delivered, while
`modelled_corpus_stats` returns the cleaned then deduplicated corpus the models
were actually trained on. Each caller selects one by name, so a page can never
silently render the other one.
"""

import re
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "input" / "spam.csv"


def clean_text(text: str) -> str:
    """Normalize raw SMS text for model ingestion.

    Args:
        text: Raw SMS message string.

    Returns:
        Cleaned, lowercased string with URL/PHONE tokens substituted.
    """
    if not isinstance(text, str):
        return ""

    # Lowercasing first so that HTTP:// is normalized before the URL patterns run
    text = text.lower()

    text = re.sub(r"https?://[^\s]+", "URL", text)
    text = re.sub(r"www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "URL", text)

    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "PHONE", text)
    text = re.sub(r"\b\d{5,}\b", "PHONE", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


@st.cache_data
def load_raw_corpus() -> pd.DataFrame:
    """Load the corpus exactly as delivered: no cleaning, no deduplication.

    This is the "raw corpus as delivered" referential. Callers that need the
    corpus the models were trained on must use `modelled_corpus_stats` instead.
    """
    df = pd.read_csv(DATA_RAW, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    return df


@st.cache_data
def modelled_corpus_stats() -> dict:
    """Rebuild the corpus the models were trained on and return its counts.

    Cleaning runs before deduplication (notebook cells 13 then 14, in that
    order). Deduplicating the raw text instead would keep the near-duplicates
    that only cleaning collapses, producing a corpus no model in this project
    was ever trained on.
    """
    df = load_raw_corpus().copy()
    n_raw = len(df)

    df["text_cleaned"] = df["text"].apply(clean_text)
    df = df.drop_duplicates(subset=["text_cleaned"], keep="first")

    return {
        "total_raw": n_raw,
        "duplicates": n_raw - len(df),
        "ham": int((df["label"] == "ham").sum()),
        "spam": int((df["label"] == "spam").sum()),
        "total_deduped": len(df),
    }
