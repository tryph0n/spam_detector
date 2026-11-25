"""Demo page -- live SMS spam prediction with selectable model (BiLSTM or TextCNN)."""

import json
import pickle
from pathlib import Path

import streamlit as st
from text_cleaning import clean_text

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

st.title("Live Demo")
if not TORCH_AVAILABLE:
    st.error(
        "The demo needs PyTorch, which is in the `train` dependency group: "
        "run `uv sync --group train`. See the README, section Train the models."
    )
    st.stop()

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SERVED_MODELS = ["BiLSTM", "TextCNN"]
HEAVY_MODELS = ["RoBERTa", "DistilBERT"]

RESULT_FILES = {
    "BiLSTM": MODELS_DIR / "bilstm_results.json",
    "TextCNN": MODELS_DIR / "textcnn_results.json",
    "RoBERTa": MODELS_DIR / "roberta_results.json",
    "DistilBERT": MODELS_DIR / "distilbert_results.json",
}

MODEL_ARTIFACTS = {
    "BiLSTM": MODELS_DIR / "lstm_baseline.pt",
    "TextCNN": MODELS_DIR / "textcnn_baseline.pt",
    "RoBERTa": MODELS_DIR / "roberta_spam" / "best_model",
    "DistilBERT": MODELS_DIR / "distilbert_spam" / "best_model",
}


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for binary text classification."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]
        dropped = self.dropout(last_hidden)
        return self.fc(dropped).squeeze(1)


class TextCNNClassifier(nn.Module):
    """TextCNN classifier following Kim (2014).

    Parallel 1D convolutions with multiple filter sizes detect n-gram features
    of different lengths. Max-over-time pooling collapses each feature map to a
    scalar, giving a fixed-size representation for any input length.

    Args:
        vocab_size: Vocabulary size including PAD and UNK tokens.
        embed_dim: Embedding dimension.
        num_filters: Number of output channels per filter size.
        filter_sizes: Tuple of kernel sizes (each corresponds to an n-gram width).
        dropout: Dropout probability applied before the final linear layer.
        padding_idx: Index reserved for padding (masked in embedding gradient).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 100,
        filter_sizes: tuple = (3, 4, 5),
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # One Conv1d per filter size -- each learns its own n-gram detectors
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Long tensor of shape (batch, seq_len) containing token indices.

        Returns:
            Float tensor of shape (batch,) -- raw logits (no sigmoid).
        """
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Conv1d expects channels-first
        # Max-over-time pooling: keep the strongest activation across positions
        conv_outs = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)


def tokenize(text, vocab, max_len):
    """Convert text to padded sequence of vocabulary indices."""
    words = str(text).lower().split()
    indices = [vocab.get(word, vocab["<UNK>"]) for word in words]
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices += [vocab["<PAD>"]] * (max_len - len(indices))
    return indices


@st.cache_resource
def load_model(model_name: str):
    """Load the requested model and shared vocabulary artifacts.

    Args:
        model_name: One of "BiLSTM" or "TextCNN".

    Returns:
        Tuple (model, vocab, max_len) or (None, vocab, max_len) if the
        checkpoint is missing (model not yet trained).
    """
    try:
        with open(MODELS_DIR / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        with open(MODELS_DIR / "tokenizer_config.pkl", "rb") as f:
            config = pickle.load(f)
    except FileNotFoundError:
        return None, None, None

    checkpoint = MODEL_ARTIFACTS[model_name]
    if not checkpoint.exists():
        return None, vocab, config["max_len"]

    if model_name == "BiLSTM":
        model = BiLSTMClassifier(vocab_size=config["vocab_size"])
    else:
        model = TextCNNClassifier(vocab_size=config["vocab_size"])

    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model, vocab, config["max_len"]


def _artifact_size(path: Path) -> int | None:
    """Return the on-disk size in bytes of a checkpoint file or a model directory."""
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return None


def _format_size(n_bytes: int) -> str:
    """Format a byte count in mebioctets, the binary unit used across the project."""
    mio = n_bytes / 1024**2
    return f"{mio:.1f} Mio" if mio < 100 else f"{mio:.0f} Mio"


def _format_params(n_params: int) -> str:
    if n_params >= 1_000_000:
        return f"{n_params / 1_000_000:.0f}M params"
    return f"{n_params / 1_000:.0f}K params"


@st.cache_data
def model_facts(model_name: str) -> str:
    """Describe a model from its own artifacts, skipping whatever is not on disk."""
    facts = []
    results_path = RESULT_FILES[model_name]
    data = {}
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)

    if "parameters" in data:
        facts.append(_format_params(data["parameters"]))
    size = _artifact_size(MODEL_ARTIFACTS[model_name])
    if size is not None:
        facts.append(_format_size(size))
    if "f1" in data:
        facts.append(f"spam F1 {data['f1']:.3f}")
    return ", ".join(facts)


def describe_models(model_names: list[str]) -> str:
    """Join model descriptions into a sentence fragment."""
    described = []
    for name in model_names:
        facts = model_facts(name)
        described.append(f"{name} ({facts})" if facts else name)
    return " and ".join(described)


def sizes_are_shown(model_names: list[str]) -> bool:
    """Report whether any listed model has an artifact on disk to measure.

    On a clone without `models/`, `describe_models` emits bare names, so the
    sentence explaining the size unit would refer to figures that are absent.
    """
    return any(
        _artifact_size(MODEL_ARTIFACTS[name]) is not None for name in model_names
    )


def predict(text, model, vocab, max_len):
    """Run prediction on a single SMS text."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned, vocab, max_len)
    tensor_in = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor_in)
        prob = torch.sigmoid(logits).item()
    return prob


st.markdown("Enter an SMS message to check if it's spam or legitimate.")

size_note = (
    "Sizes are measured on disk in mebioctets (1 Mio = 1024 x 1024 bytes). "
    if sizes_are_shown(SERVED_MODELS + HEAVY_MODELS)
    else ""
)

st.info(
    f"This demo serves {describe_models(SERVED_MODELS)}, the two lightweight models "
    f"the exported vocabulary can load. {describe_models(HEAVY_MODELS)} score higher, "
    "but the app does not depend on `transformers` at all and their checkpoints are "
    "two orders of magnitude larger, so neither one is served here. The GAN-augmented "
    "BiLSTM is just as small, but its training run rebuilt its own 3,654-token "
    "vocabulary instead of reusing the 3,547-token one used here, so its "
    "checkpoint cannot be reloaded against it. "
    f"{size_note}"
    "Full results are available on the Results page."
)

model_choice = st.radio("Select model", SERVED_MODELS, horizontal=True)

model, vocab, max_len = load_model(model_choice)

if vocab is None:
    st.info(
        "No trained model in models/. The models are not included in this "
        "repository: train them first with `uv sync --group train`, then run all "
        "cells of `spam-detection_DL.ipynb` (about 3 hours on CPU). See the README, "
        "section Train the models."
    )
    st.stop()

if model is None:
    st.warning(
        f"{model_choice} model not trained yet. "
        "See the README, section Train the models."
    )
    st.stop()

EXAMPLE_TEXTS = [
    ("Ham", "Hey, are you coming to dinner tonight?"),
    ("Ham", "Ok I'll be there in 10 min"),
    ("Spam", "WINNER! You have been selected for a free prize! Call 09061234567 NOW!"),
    (
        "Spam",
        "Congratulations! You've won a $1000 gift card. Click here to claim: http://scam.link",
    ),
    (
        "Ham",
        "Meeting confirmed for 3pm in the main conference room. Bring your laptop.",
    ),
    (
        "Ham",
        "Mom says hi and wants to know if you're coming for Sunday lunch. Let me know so she can plan.",
    ),
    ("Ham", "Can you pick up some milk on your way home? We're also out of bread."),
    (
        "Spam",
        "URGENT! Your bank account has been compromised. Click here to verify your identity immediately: http://secure-bank.fake",
    ),
    (
        "Spam",
        "You have been chosen to receive a FREE iPhone 15! Text CLAIM to 08001234567 before midnight to collect your prize!",
    ),
    (
        "Spam",
        "Final notice: your tax refund of $3,479.50 is pending. Reply with your details or call 09058726435 NOW to avoid forfeiture.",
    ),
]

# Labels are derived from the texts so the dropdown cannot announce one message
# and load another
EXAMPLES = {"(Select an example)": ""} | {
    f"{kind}: {text}": text for kind, text in EXAMPLE_TEXTS
}

selected = st.selectbox("Try an example", options=list(EXAMPLES.keys()))
example_text = EXAMPLES[selected]

user_input = st.text_area(
    "SMS message",
    value=example_text,
    height=100,
    placeholder="Type or paste an SMS message here...",
)

if st.button("Analyze", type="primary", disabled=not user_input):
    prob = predict(user_input, model, vocab, max_len)
    is_spam = prob > 0.5
    confidence = prob if is_spam else 1 - prob

    st.divider()

    col1, col2 = st.columns(2)
    if is_spam:
        col1.error("**SPAM** detected")
    else:
        col1.success("**HAM** (legitimate)")
    col2.metric("Confidence", f"{confidence:.1%}")

    st.progress(prob, text=f"Spam probability: {prob:.1%}")

    with st.expander("Details"):
        cleaned = clean_text(user_input)
        st.markdown(f"**Cleaned text:** {cleaned}")
        st.markdown(f"**Spam probability (sigmoid output):** {prob:.6f}")
        st.markdown(f"**Model:** {describe_models([model_choice])}")
