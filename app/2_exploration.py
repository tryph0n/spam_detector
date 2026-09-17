"""Exploration page -- EDA on the raw delivered corpus, before cleaning and dedup."""

from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from text_cleaning import ROOT, load_raw_corpus, modelled_corpus_stats

COLORS = {"ham": "#2ecc71", "spam": "#e74c3c"}

TRAIN_SPLIT = ROOT / "data" / "processed" / "train.csv"

RAW_CORPUS = "raw corpus"


@st.cache_data
def load_data():
    """Add length features to the raw corpus -- this page describes the file as delivered."""
    df = load_raw_corpus().copy()
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    return df


@st.cache_data
def load_train_balance() -> dict | None:
    """Return the ham/spam counts of the training split, or None if absent."""
    if not TRAIN_SPLIT.exists():
        return None
    df = pd.read_csv(TRAIN_SPLIT)
    return {
        "ham": int((df["label"] == 0).sum()),
        "spam": int((df["label"] == 1).sum()),
    }


def plot_class_distribution(df):
    """Plot class distribution bar chart and pie chart."""
    class_counts = df["label"].value_counts().reset_index()
    class_counts.columns = ["label", "count"]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        subplot_titles=[
            f"Count -- {RAW_CORPUS}, {len(df):,} messages",
            f"Share -- {RAW_CORPUS}",
        ],
    )

    fig.add_trace(
        go.Bar(
            x=class_counts["label"],
            y=class_counts["count"],
            marker_color=[COLORS.get(name, "#999") for name in class_counts["label"]],
            text=class_counts["count"],
            textposition="inside",
            textfont=dict(size=16, color="white"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Pie(
            labels=class_counts["label"],
            values=class_counts["count"],
            marker_colors=[COLORS.get(name, "#999") for name in class_counts["label"]],
            textinfo="label+percent",
            pull=[0.05, 0.05],
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=400, showlegend=False)
    return fig


def plot_length_distribution(df, log_scale=False):
    """Plot message length histograms."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Character count -- {RAW_CORPUS}, {len(df):,} messages",
            f"Word count -- {RAW_CORPUS}, {len(df):,} messages",
        ],
    )

    for label, color in COLORS.items():
        subset = df[df["label"] == label]
        fig.add_trace(
            go.Histogram(
                x=subset["char_count"],
                name=label.capitalize(),
                marker_color=color,
                opacity=0.6,
                nbinsx=50,
                legendgroup=label,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=subset["word_count"],
                name=label.capitalize(),
                marker_color=color,
                opacity=0.6,
                nbinsx=50,
                legendgroup=label,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(barmode="overlay", height=450, legend=dict(x=0.01, y=0.99))
    fig.update_xaxes(title_text="Character Count", row=1, col=1)
    fig.update_xaxes(title_text="Word Count", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    if log_scale:
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=2)
    return fig


def plot_word_frequency(df):
    """Plot top words for ham vs spam."""
    n_spam = int((df["label"] == "spam").sum())
    n_ham = int((df["label"] == "ham").sum())

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Top 15 words in spam -- {RAW_CORPUS}, {n_spam:,} messages",
            f"Top 15 words in ham -- {RAW_CORPUS}, {n_ham:,} messages",
        ],
    )

    for idx, (label, color) in enumerate(
        [("spam", COLORS["spam"]), ("ham", COLORS["ham"])]
    ):
        texts = df[df["label"] == label]["text"]
        all_words = []
        for text in texts:
            all_words.extend(str(text).lower().split())
        freq = Counter(all_words).most_common(15)
        words, counts = zip(*freq, strict=True)

        fig.add_trace(
            go.Bar(
                y=list(words),
                x=list(counts),
                orientation="h",
                marker_color=color,
                name=label.capitalize(),
                showlegend=False,
            ),
            row=1,
            col=idx + 1,
        )

    fig.update_layout(height=500)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Frequency")
    return fig


def plot_exclusive_words(df):
    """Plot words that appear exclusively in spam or ham."""
    spam_words = set()
    ham_words = set()
    for _, row in df.iterrows():
        words = set(str(row["text"]).lower().split())
        if row["label"] == "spam":
            spam_words.update(words)
        else:
            ham_words.update(words)

    only_spam = spam_words - ham_words
    only_ham = ham_words - spam_words

    spam_counts = Counter()
    ham_counts = Counter()
    for _, row in df.iterrows():
        words = str(row["text"]).lower().split()
        if row["label"] == "spam":
            for w in words:
                if w in only_spam:
                    spam_counts[w] += 1
        else:
            for w in words:
                if w in only_ham:
                    ham_counts[w] += 1

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Exclusive to spam -- {RAW_CORPUS}, {len(only_spam):,} unique words",
            f"Exclusive to ham -- {RAW_CORPUS}, {len(only_ham):,} unique words",
        ],
    )

    for idx, (counts, color) in enumerate(
        [
            (spam_counts.most_common(15), COLORS["spam"]),
            (ham_counts.most_common(15), COLORS["ham"]),
        ]
    ):
        if counts:
            words, freqs = zip(*counts, strict=True)
            fig.add_trace(
                go.Bar(
                    y=list(words),
                    x=list(freqs),
                    orientation="h",
                    marker_color=color,
                    showlegend=False,
                ),
                row=1,
                col=idx + 1,
            )

    fig.update_layout(height=500)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Frequency")
    return fig


st.title("Data Exploration")

df = load_data()
modelled = modelled_corpus_stats()

st.warning(
    f"Referential of this page: the **raw corpus as delivered**, "
    f"{len(df):,} messages, before cleaning and deduplication. Every chart below "
    f"describes that file. The corpus the models were trained and evaluated on is "
    f"smaller -- {modelled['total_deduped']:,} messages ({modelled['ham']:,} ham / "
    f"{modelled['spam']:,} spam) -- and is the one shown on the Accueil page. "
    f"The two never coincide: figures from this page must not be quoted as "
    f"properties of the modelled dataset."
)

st.header(f"Class Distribution ({RAW_CORPUS})")
st.plotly_chart(plot_class_distribution(df), width="stretch")

n_ham_raw = int((df["label"] == "ham").sum())
n_spam_raw = int((df["label"] == "spam").sum())
train = load_train_balance()
train_line = (
    "The training loss did not use this ratio: `pos_weight` was computed on the "
    f"training split of the cleaned deduplicated corpus, {train['ham']:,} ham / "
    f"{train['spam']:,} spam = {train['ham'] / train['spam']:.3f}."
    if train
    else "The training loss used a `pos_weight` computed on the training split of "
    "the cleaned deduplicated corpus, not on this ratio."
)

st.markdown(
    f"**Imbalance ratio on the {RAW_CORPUS}:** "
    f"{n_ham_raw / n_spam_raw:.2f}:1 (ham:spam), "
    f"{n_ham_raw:,} ham / {n_spam_raw:,} spam. {train_line}"
)

st.header(f"Message Length ({RAW_CORPUS})")
log_scale = st.toggle("Log scale (Y axis)")
st.plotly_chart(plot_length_distribution(df, log_scale=log_scale), width="stretch")

col1, col2 = st.columns(2)
for col, label in zip([col1, col2], ["ham", "spam"], strict=True):
    subset = df[df["label"] == label]
    col.markdown(
        f"**{label.capitalize()}** ({RAW_CORPUS}) -- "
        f"avg chars: {subset['char_count'].mean():.0f}, "
        f"avg words: {subset['word_count'].mean():.0f}"
    )

st.caption(
    "Word counts here and in the histogram above split the raw text on whitespace, "
    "with no stopword removal and no URL/phone substitution -- punctuation stays "
    "attached to the token it touches, unlike the text the models ingest."
)

st.header(f"Most Frequent Words ({RAW_CORPUS})")
st.plotly_chart(plot_word_frequency(df), width="stretch")
st.caption(
    "Counted on the raw text, lowercased and split on whitespace, with no stopword "
    "removal and no URL/phone substitution -- unlike the text the models ingest."
)

st.header(f"Words Exclusive to Each Class ({RAW_CORPUS})")
st.plotly_chart(plot_exclusive_words(df), width="stretch")
st.caption(
    "A word is counted as exclusive when it never appears in the other class "
    "anywhere in the raw corpus. Words are lowercased whitespace tokens of the raw "
    "text, so the unique-word counts in the titles follow that same convention."
)
