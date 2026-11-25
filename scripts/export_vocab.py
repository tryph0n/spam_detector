"""Export the vocabulary and tokenizer config from the training split.

Reproduces the vocabulary step of `spam-detection_DL.ipynb`, so that the
BiLSTM and TextCNN checkpoints can be loaded for inference by the Demo page.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from text_cleaning import clean_text  # noqa: E402

logger = logging.getLogger(__name__)


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "models",
        help="folder for vocab.pkl and tokenizer_config.pkl (default: models/)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    train_df["clean_text"] = train_df["text"].apply(clean_text)

    vocab = build_vocabulary(train_df["clean_text"], min_freq=2)

    # 95th percentile of word counts, as in the notebook
    train_lengths = train_df["clean_text"].apply(lambda x: len(str(x).split()))
    max_len = int(np.percentile(train_lengths, 95))

    with open(out_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    tokenizer_config = {
        "max_len": max_len,
        "vocab_size": len(vocab),
    }
    with open(out_dir / "tokenizer_config.pkl", "wb") as f:
        pickle.dump(tokenizer_config, f)

    logger.info("Vocabulary size: %d", len(vocab))
    logger.info("Max sequence length: %d", max_len)
    logger.info("Saved: %s", out_dir / "vocab.pkl")
    logger.info("Saved: %s", out_dir / "tokenizer_config.pkl")


if __name__ == "__main__":
    main()
