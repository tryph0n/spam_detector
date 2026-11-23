# SMS Spam Detector

[![CI](https://github.com/tryph0n/spam_detector/actions/workflows/ci.yml/badge.svg)](https://github.com/tryph0n/spam_detector/actions/workflows/ci.yml)

Fictional business case from a data science training course: a telecom operator (AT&T in the course brief) wants to flag spam SMS automatically, based only on the message text. No affiliation with AT&T.

Deep learning SMS spam classifier comparing five model variants -- two trained from scratch, two fine-tuned transformers, and one GAN-augmented rerun -- with a Streamlit web app for live predictions.

## Results

Test set: 772 messages (678 ham, 94 spam), held out from the 5,142-message modelled corpus described below. Spam is the positive class, so every precision/recall/F1 figure is a spam-class metric. All training times are CPU times -- no GPU was available for this run.

| Model | Params | Accuracy | F1 (spam) | Training time |
|-------|--------|----------|------|---------------|
| BiLSTM | 553K | 0.959 | 0.845 | 679s |
| TextCNN | 608K | 0.977 | 0.902 | 221s |
| RoBERTa | 125M | 0.992 | 0.968 | 4060s |
| DistilBERT | 67M | 0.991 | 0.962 | 2057s |
| BiLSTM GAN-Augmented | 567K | 0.972 | 0.879 | 2764s |

RoBERTa achieves the best spam F1 (0.968). DistilBERT is nearly equivalent (0.962) at half the training time and 46% fewer parameters.

`BiLSTM GAN-Augmented` is the same BiLSTM retrained on a training set rebalanced with GAN-generated spam embeddings. Those embeddings are mapped back to text by nearest-neighbour retrieval, so the augmentation duplicates existing spam messages rather than creating new ones: precision rises, recall falls.

Read accuracy against the trivial baseline: always predicting ham scores 0.878 on this test set, so accuracy alone says little here. With only 94 spam messages in the test set, a single error moves recall by about 1.1 points and no confidence interval was computed, so the 0.006 F1 gap between RoBERTa and DistilBERT is not a demonstrated difference.

## Installation

Prerequisites: Python 3.11 (pinned in `.python-version`) and [uv](https://github.com/astral-sh/uv).

### Setup

```bash
git clone https://github.com/tryph0n/spam_detector.git
cd spam_detector
uv sync
```

`uv sync` installs the app (Streamlit, pandas, Plotly) and the `dev` group (`ruff`, `pytest`). It does not install PyTorch: the training libraries are in the `train` group, see [Train the models](#train-the-models).

### Teardown

```bash
rm -rf .venv models/
```

Deleting `models/` is irreversible from the repository: nothing under it is tracked in git. The trained checkpoints can only come back from a full notebook run, see [Train the models](#train-the-models). The vocabulary files are the exception: `uv run python scripts/export_vocab.py` rebuilds `vocab.pkl` and `tokenizer_config.pkl` identically (3,547 entries, `max_len` 32) from the tracked `data/processed/train.csv`. It creates `models/` if it is missing; `--out-dir` writes the two files to another folder.

## Train the models

The trained models are not included in this repository. Train them before you use the Resultats and Demo pages:

```bash
uv sync --group train
uv run jupyter notebook spam-detection_DL.ipynb
```

`uv sync --group train` adds the `train` dependency group: PyTorch (CPU build), Transformers, sentence-transformers and Jupyter. In Jupyter, run all cells of `spam-detection_DL.ipynb`. A later plain `uv sync` removes the `train` group again, so keep `--group train` while you train or use the Demo page.

Cost of a full run, as recorded rather than estimated:

- **3h07 wall clock on CPU** for the committed run, read from the notebook's own log timestamps (10:03:28 to 13:10:41).
- **2h43 of that is model training**, the sum of the five durations the notebook measures and writes to `models/*_results.json` (679 + 221 + 4060 + 2057 + 2764 s).
- The GAN generator's own training time **is never measured**. It can only be inferred from the surrounding log timestamps, at roughly 21 minutes, and should be treated as an estimate rather than a figure.

What the run needs and changes:

- **Internet access**: the notebook downloads `roberta-base`, `distilbert-base-uncased` and the sentence-transformers model `all-MiniLM-L6-v2` from the Hugging Face Hub.
- **Disk space**: the run writes about 5 GB to `models/`. The app uses 9 of those files (4.5 MiB): the five `*_results.json` files, `vocab.pkl`, `tokenizer_config.pkl`, `lstm_baseline.pt` and `textcnn_baseline.pt`.
- **Tracked files**: the run rewrites the tracked `plots/*.png` and `data/processed/*.csv`. To restore the committed versions: `git checkout -- plots data/processed`.

## Usage

### Streamlit App

```bash
uv run streamlit run app/main.py
```

Four pages: Accueil (overview), Exploration (EDA), Resultats (model comparison), Demo (live prediction).

Launch the app through `app/main.py` only. Three of the four pages -- Accueil, Exploration and Demo -- import `app/text_cleaning.py` as a sibling module, which resolves because `streamlit run` puts the entry script's directory on `sys.path`. Loading one of those three through `streamlit.testing.v1.AppTest` raises `ModuleNotFoundError: No module named 'text_cleaning'` unless `app/` is added to `sys.path` first; Resultats imports no sibling module, builds its own `ROOT` from `__file__`, and renders under AppTest with no such patch. Running `python app/1_accueil.py` directly does resolve the import -- CPython puts the script's own directory on `sys.path` -- but it exits without rendering anything, there being no Streamlit runtime.

**A fresh clone cannot run the full app.** `models/` is excluded by `.gitignore` and no file under it is tracked, so a clone contains no trained artifacts. Accueil and Exploration work out of the box because they read `data/`, which is tracked. Resultats and Demo show a message that points to [Train the models](#train-the-models); the Demo page also needs the `train` group, because it loads PyTorch.

### Notebook

```bash
uv run jupyter notebook spam-detection_DL.ipynb
```

`spam-detection_DL.ipynb` is the deliverable: it runs the whole pipeline, from the raw CSV to the five trained models, and writes everything under `models/` and `plots/`. It needs the `train` group, see [Train the models](#train-the-models). The other notebooks in `notebooks/` are earlier exploratory drafts, superseded by this one. `01_eda.ipynb` and `02_preprocessing.ipynb` carry their stored outputs; `03_model_simple.ipynb` and `04_transfer_learning.ipynb` were never executed -- zero executed cells and zero stored outputs -- so their prose states intent, not a recorded result.

## Dataset

`data/input/spam.csv`: SMS Spam Collection, Almeida and Hidalgo (2011), UCI Machine Learning Repository, DOI 10.24432/C5CC84, license CC BY 4.0. This copy uses the Kaggle CSV layout (v1, v2; Latin-1) and was provided by the Jedha data science training: https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Deep+Learning/project/spam.csv. It holds real SMS messages, some with phone numbers; it is kept unchanged for reproducibility.

- **As delivered:** 5,572 raw rows -- 4,825 ham / 747 spam (86.6% / 13.4%).
- **As modelled:** 5,142 messages -- 4,515 ham / 627 spam (87.8% / 12.2%). The pipeline lowercases, replaces URLs and phone-number patterns with `URL`/`PHONE` tokens, then deduplicates **on the cleaned text**, which removes 430 rows (7.72%). Deduplicating the raw text instead removes only 403 rows and yields a different corpus that no model here was ever trained on.
- **Splits:** stratified 70/15/15 with `random_state=42` -- train 3,599 / val 771 / test 772, summing to 5,142, with zero text overlap between splits.

Every figure in the Results table was measured on the 5,142-message corpus, never on the raw 5,572.

## Tech Stack

| Category | Technology |
|----------|------------|
| Deep Learning | PyTorch, HuggingFace Transformers |
| NLP | scikit-learn, sentence-transformers |
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web App | Streamlit |
| Package Manager | uv |

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

ruff skips the notebooks (`extend-exclude` in `pyproject.toml`): their code stays the code that produced the recorded results. The tests cover `clean_text` and its parity with the notebook version. CI runs these three commands on each pull request and each push to `main`, and checks that PyTorch is not installed.

## License

The code is under the MIT license, see `LICENSE`. The dataset is under CC BY 4.0, see [Dataset](#dataset).
