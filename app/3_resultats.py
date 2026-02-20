"""Results page -- dynamic model comparison loaded from JSON files."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_FILES = {
    "BiLSTM": "bilstm_results.json",
    "TextCNN": "textcnn_results.json",
    "RoBERTa": "roberta_results.json",
    "DistilBERT": "distilbert_results.json",
    "BiLSTM GAN-Augmented": "bilstm_augmented_results.json",
}

MODEL_COLORS = {
    "BiLSTM": "#2E86AB",
    "TextCNN": "#A23B72",
    "RoBERTa": "#F18F01",
    "DistilBERT": "#C73E1D",
    "BiLSTM GAN-Augmented": "#3B1F2B",
}


@st.cache_data
def load_results() -> dict:
    results = {}
    for name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
        else:
            st.warning(f"Model {name} results not found: {path}")
    return results


def build_performance_table(results: dict) -> pd.DataFrame:
    rows = []
    for model, data in results.items():
        rows.append({
            "Model": model,
            "Parameters": f"{data['parameters']:,}",
            "Train Time (s)": f"{data['training_time_s']:.1f}",
            "Accuracy": f"{data['accuracy']:.4f}",
            "Precision (Spam)": f"{data['precision']:.4f}",
            "Recall (Spam)": f"{data['recall']:.4f}",
            "F1 (Spam)": f"{data['f1']:.4f}",
        })
    return pd.DataFrame(rows)


def plot_metrics_comparison(results: dict) -> go.Figure:
    metric_labels = ["Accuracy", "Precision (Spam)", "Recall (Spam)", "F1 (Spam)"]
    metric_keys = ["accuracy", "precision", "recall", "f1"]

    fig = go.Figure()
    for model, data in results.items():
        values = [data[k] for k in metric_keys]
        fig.add_trace(go.Bar(
            name=model,
            x=metric_labels,
            y=values,
            marker_color=MODEL_COLORS.get(model, "#888888"),
            opacity=0.85,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))

    all_values = [data[k] for data in results.values() for k in metric_keys]
    y_min = max(0, min(all_values) - 0.005)

    fig.update_layout(
        barmode="group",
        title=dict(text="Performance Metrics Comparison", font=dict(size=16)),
        yaxis=dict(title="Score", range=[y_min, 1.005]),
    )
    return fig


def render_confusion_matrices(results: dict) -> None:
    """Render confusion matrices in a 2x2 grid, excluding augmented variants."""
    main_models = ["BiLSTM", "TextCNN", "RoBERTa", "DistilBERT"]
    available = [m for m in main_models if m in results]
    labels = ["Ham", "Spam"]

    for i in range(0, len(available), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(available):
                break
            model = available[idx]
            cm = results[model]["confusion_matrix"]
            fp = cm[0][1]

            annotations_text = [[str(cm[r][c]) for c in range(2)] for r in range(2)]

            fig = go.Figure(go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                showscale=False,
                text=annotations_text,
                texttemplate="%{text}",
                textfont=dict(size=18, color="black"),
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            ))

            # Flag false positives -- legitimate messages blocked by the classifier
            fig.add_annotation(
                x="Spam",
                y="Ham",
                text=f"FP: {fp}",
                showarrow=False,
                font=dict(color="red", size=12, family="Arial Black"),
                yshift=-25,
            )

            fig.update_layout(
                title=dict(text=model, font=dict(size=14)),
                xaxis_title="Predicted",
                yaxis_title="True Label",
                height=350,
            )

            with col:
                st.plotly_chart(fig, width='stretch')


def render_gan_impact(results: dict) -> None:
    st.header("GAN Augmentation Impact")

    if "BiLSTM" not in results or "BiLSTM GAN-Augmented" not in results:
        st.info("GAN augmentation results not available yet.")
        return

    base = results["BiLSTM"]
    aug = results["BiLSTM GAN-Augmented"]

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
    ]

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("BiLSTM (baseline)")
        for label, key in metrics:
            st.metric(label=label, value=f"{base[key]:.4f}")
    with col_right:
        st.subheader("BiLSTM GAN-Augmented")
        for label, key in metrics:
            delta = aug[key] - base[key]
            st.metric(label=label, value=f"{aug[key]:.4f}", delta=f"{delta:+.4f}")

    deltas = {key: aug[key] - base[key] for _, key in metrics}
    improved = [label for label, key in metrics if deltas[key] > 0]
    regressed = [label for label, key in metrics if deltas[key] < 0]
    neutral = [label for label, key in metrics if deltas[key] == 0]

    if improved and not regressed:
        summary = (
            f"GAN augmentation improved all tracked metrics "
            f"({', '.join(improved)}), confirming that synthetic oversampling "
            f"of the minority class adds useful signal."
        )
    elif regressed and not improved:
        summary = (
            f"GAN augmentation degraded all tracked metrics "
            f"({', '.join(regressed)}), suggesting the synthetic samples "
            f"introduce noise rather than useful signal."
        )
    else:
        improved_str = f"improved {', '.join(improved)}" if improved else ""
        regressed_str = f"degraded {', '.join(regressed)}" if regressed else ""
        neutral_str = f"left {', '.join(neutral)} unchanged" if neutral else ""
        parts = [p for p in [improved_str, regressed_str, neutral_str] if p]
        summary = (
            f"GAN augmentation had mixed results: it {', and '.join(parts)}. "
            f"Consider re-tuning the GAN or filtering low-quality synthetic samples."
        )

    st.markdown(f"**Interpretation:** {summary}")


def render_analysis(results: dict) -> None:
    best_f1_model = max(results, key=lambda m: results[m]["f1"])
    best_recall_model = max(results, key=lambda m: results[m]["recall"])
    fastest_model = min(results, key=lambda m: results[m]["training_time_s"])
    smallest_model = min(results, key=lambda m: results[m]["parameters"])

    st.markdown(f"""
### Key Findings

- **Best F1 score**: {best_f1_model} ({results[best_f1_model]['f1']:.4f})
- **Best spam recall**: {best_recall_model} ({results[best_recall_model]['recall']:.4f})
- **Fastest training**: {fastest_model} ({results[fastest_model]['training_time_s']:.1f}s)
- **Fewest parameters**: {smallest_model} ({results[smallest_model]['parameters']:,})

**Business priority: minimize false positives** (blocking legitimate messages is worse than
letting some spam through). Models with the highest precision reduce the risk of incorrectly
flagging ham messages.
""")


st.title("Model Results")

results = load_results()

if not results:
    st.error("No model results found. Run training notebooks first.")
    st.stop()

st.header("Performance Comparison")
st.dataframe(build_performance_table(results), width='stretch', hide_index=True)

st.header("Metrics Comparison")
st.plotly_chart(plot_metrics_comparison(results), width='stretch')

st.header("Confusion Matrices")
st.caption("FP = False Positives (legitimate SMS classified as spam)")
render_confusion_matrices(results)

st.header("Analysis")
render_analysis(results)

render_gan_impact(results)
