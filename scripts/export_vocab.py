"""Export vocabulary and tokenizer config from training data.

Reproduces the exact vocabulary building process from notebook 03
so that the BiLSTM model can be loaded for inference.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def clean_text(text):
    """Minimal cleaning matching notebook 02 preprocessing."""
    import re

    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
        r"[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "URL",
        text,
    )
    text = re.sub(r"www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "URL", text)
    text = re.sub(
        r"\b\d{5,}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "PHONE", text
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocabulary(texts, min_freq=2):
    """Build word-to-index mapping from text corpus."""
    word_counts = {}
    for text in texts:
        for word in str(text).lower().split():
            word_counts[word] = word_counts.get(word, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


def main():
    data_dir = Path("data/processed")
    model_dir = Path("models")

    train_df = pd.read_csv(data_dir / "train.csv")

    # Clean text (same pipeline as notebook 02)
    train_df["clean_text"] = train_df["text"].apply(clean_text)

    # Build vocabulary (same as notebook 03)
    vocab = build_vocabulary(train_df["clean_text"], min_freq=2)

    # Compute max_len (95th percentile of word counts)
    train_lengths = train_df["clean_text"].apply(
        lambda x: len(str(x).split())
    )
    max_len = int(np.percentile(train_lengths, 95))

    # Save artifacts
    with open(model_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    tokenizer_config = {
        "max_len": max_len,
        "vocab_size": len(vocab),
    }
    with open(model_dir / "tokenizer_config.pkl", "wb") as f:
        pickle.dump(tokenizer_config, f)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Max sequence length: {max_len}")
    print(f"Saved: {model_dir / 'vocab.pkl'}")
    print(f"Saved: {model_dir / 'tokenizer_config.pkl'}")


if __name__ == "__main__":
    main()
