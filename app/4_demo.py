"""Demo page -- live SMS spam prediction with selectable model (BiLSTM or TextCNN)."""

import json
import pickle
import re
from pathlib import Path

import streamlit as st

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

RESULT_FILES = {
    "BiLSTM": MODELS_DIR / "bilstm_results.json",
    "TextCNN": MODELS_DIR / "textcnn_results.json",
}

MODEL_CHECKPOINTS = {
    "BiLSTM": MODELS_DIR / "lstm_baseline.pt",
    "TextCNN": MODELS_DIR / "textcnn_baseline.pt",
}


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for binary text classification."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
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
        # One Conv1d per filter size — each learns its own n-gram detectors
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
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
        x = x.permute(0, 2, 1)         # Conv1d expects channels-first
        # Max-over-time pooling: keep the strongest activation across positions
        conv_outs = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)


def clean_text(text):
    """Clean text for model input."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "URL", text,
    )
    text = re.sub(r"www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "URL", text)
    text = re.sub(r"\b\d{5,}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "PHONE", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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

    checkpoint = MODEL_CHECKPOINTS[model_name]
    if not checkpoint.exists():
        return None, vocab, config["max_len"]

    if model_name == "BiLSTM":
        model = BiLSTMClassifier(vocab_size=config["vocab_size"])
    else:
        model = TextCNNClassifier(vocab_size=config["vocab_size"])

    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model, vocab, config["max_len"]


def _model_label(model_name: str) -> str:
    """Build a display label with parameter count from results JSON if available."""
    results_path = RESULT_FILES[model_name]
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        n_params = data.get("parameters")
        if n_params is not None:
            return f"{model_name} ({n_params:,} parameters)"
    return model_name


def predict(text, model, vocab, max_len):
    """Run prediction on a single SMS text."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned, vocab, max_len)
    tensor_in = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor_in)
        prob = torch.sigmoid(logits).item()
    return prob


st.title("Live Demo")

if not TORCH_AVAILABLE:
    st.error(
        "PyTorch is not installed. Install it with: `uv sync` "
        "or `uv pip install torch`"
    )
    st.stop()

st.markdown("Enter an SMS message to check if it's spam or legitimate.")

st.info(
    "This demo features BiLSTM (553K params, 2 MB) and TextCNN (608K params, 2 MB), "
    "the two lightweight models of the project. RoBERTa (125M params, 481 MB) and "
    "DistilBERT (67M params, 257 MB) achieve higher F1 scores (0.97/0.96) but their "
    "size and CPU inference time make them impractical for an interactive demo. "
    "Full results are available on the Results page."
)

model_choice = st.radio("Select model", ["BiLSTM", "TextCNN"], horizontal=True)

model, vocab, max_len = load_model(model_choice)

if vocab is None:
    st.error("Vocabulary not found. Run the notebook first.")
    st.stop()

if model is None:
    st.warning(
        f"{model_choice} model not trained yet. "
        "Run the notebook first to generate the checkpoint."
    )
    st.stop()

EXAMPLES = {
    "(Select an example)": "",
    "Ham: Hey, are you coming to dinner tonight?": "Hey, are you coming to dinner tonight?",
    "Ham: Ok I'll be there in 10 min": "Ok I'll be there in 10 min",
    "Spam: WINNER! You have been selected for a free prize! Call 09061234567 NOW!": "WINNER! You have been selected for a free prize! Call 09061234567 NOW!",
    "Spam: Congratulations! You've won a $1000 gift card. Click here: http://scam.link": "Congratulations! You've won a $1000 gift card. Click here to claim: http://scam.link",
    "Ham: Meeting confirmed for 3pm in the main conference room": "Meeting confirmed for 3pm in the main conference room. Bring your laptop.",
    "Ham: Mom says hi and wants to know if you're coming for Sunday lunch": "Mom says hi and wants to know if you're coming for Sunday lunch. Let me know so she can plan.",
    "Ham: Can you pick up some milk on your way home?": "Can you pick up some milk on your way home? We're also out of bread.",
    "Spam: URGENT! Your bank account has been compromised. Verify now: http://secure-bank.fake": "URGENT! Your bank account has been compromised. Click here to verify your identity immediately: http://secure-bank.fake",
    "Spam: You have been chosen to receive a FREE iPhone 15! Claim before midnight: 08001234567": "You have been chosen to receive a FREE iPhone 15! Text CLAIM to 08001234567 before midnight to collect your prize!",
    "Spam: Final notice: your tax refund of $3,479.50 is pending. Act now to avoid forfeiture": "Final notice: your tax refund of $3,479.50 is pending. Reply with your details or call 09058726435 NOW to avoid forfeiture.",
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
        st.markdown(f"**Raw logit probability:** {prob:.6f}")
        st.markdown(f"**Model:** {_model_label(model_choice)}")
