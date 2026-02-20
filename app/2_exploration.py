"""Exploration page -- EDA visualizations."""

from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

COLORS = {"ham": "#2ecc71", "spam": "#e74c3c"}


@st.cache_data
def load_data():
    """Load and prepare data for EDA charts."""
    df = pd.read_csv(Path("data/input/spam.csv"), encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    return df


def plot_class_distribution(df):
    """Plot class distribution bar chart and pie chart."""
    class_counts = df["label"].value_counts().reset_index()
    class_counts.columns = ["label", "count"]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        subplot_titles=["Class Distribution (Count)", "Class Distribution (%)"],
    )

    fig.add_trace(
        go.Bar(
            x=class_counts["label"], y=class_counts["count"],
            marker_color=[COLORS.get(l, "#999") for l in class_counts["label"]],
            text=class_counts["count"], textposition="inside",
            textfont=dict(size=16, color="white"),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Pie(
            labels=class_counts["label"], values=class_counts["count"],
            marker_colors=[COLORS.get(l, "#999") for l in class_counts["label"]],
            textinfo="label+percent", pull=[0.05, 0.05],
        ),
        row=1, col=2,
    )

    fig.update_layout(height=400, showlegend=False)
    return fig


def plot_length_distribution(df, log_scale=False):
    """Plot message length histograms."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Character Count Distribution", "Word Count Distribution"],
    )

    for label, color in COLORS.items():
        subset = df[df["label"] == label]
        fig.add_trace(
            go.Histogram(
                x=subset["char_count"], name=label.capitalize(),
                marker_color=color, opacity=0.6, nbinsx=50,
                legendgroup=label,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=subset["word_count"], name=label.capitalize(),
                marker_color=color, opacity=0.6, nbinsx=50,
                legendgroup=label, showlegend=False,
            ),
            row=1, col=2,
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
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Top 15 Words in Spam", "Top 15 Words in Ham"],
    )

    for idx, (label, color) in enumerate([("spam", COLORS["spam"]), ("ham", COLORS["ham"])]):
        texts = df[df["label"] == label]["text"]
        all_words = []
        for text in texts:
            all_words.extend(str(text).lower().split())
        freq = Counter(all_words).most_common(15)
        words, counts = zip(*freq)

        fig.add_trace(
            go.Bar(
                y=list(words), x=list(counts),
                orientation="h", marker_color=color,
                name=label.capitalize(), showlegend=False,
            ),
            row=1, col=idx + 1,
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

    # Count frequencies of exclusive words
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
        rows=1, cols=2,
        subplot_titles=[
            f"Top 15 Words Exclusive to Spam ({len(only_spam)} unique)",
            f"Top 15 Words Exclusive to Ham ({len(only_ham)} unique)",
        ],
    )

    for idx, (counts, color) in enumerate([
        (spam_counts.most_common(15), COLORS["spam"]),
        (ham_counts.most_common(15), COLORS["ham"]),
    ]):
        if counts:
            words, freqs = zip(*counts)
            fig.add_trace(
                go.Bar(
                    y=list(words), x=list(freqs),
                    orientation="h", marker_color=color,
                    showlegend=False,
                ),
                row=1, col=idx + 1,
            )

    fig.update_layout(height=500)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Frequency")
    return fig


# --- Page ---
st.title("Data Exploration")

df = load_data()

st.header("Class Distribution")
st.plotly_chart(plot_class_distribution(df), width='stretch')

st.markdown(
    f"""
    **Imbalance ratio:** {len(df[df['label'] == 'ham']) / len(df[df['label'] == 'spam']):.1f}:1
    (ham:spam). This imbalance is handled via weighted loss during training.
    """
)

st.header("Message Length")
log_scale = st.toggle("Log scale (Y axis)")
st.plotly_chart(plot_length_distribution(df, log_scale=log_scale), width='stretch')

col1, col2 = st.columns(2)
for col, label in zip([col1, col2], ["ham", "spam"]):
    subset = df[df["label"] == label]
    col.markdown(f"**{label.capitalize()}** -- Avg chars: {subset['char_count'].mean():.0f}, "
                 f"Avg words: {subset['word_count'].mean():.0f}")

st.header("Most Frequent Words")
st.plotly_chart(plot_word_frequency(df), width='stretch')

st.header("Words Exclusive to Each Class")
st.plotly_chart(plot_exclusive_words(df), width='stretch')
