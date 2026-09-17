# SMS Spam Detector

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

Prerequisites: Python 3.11 or 3.12 (`requires-python = ">=3.11,<3.13"`) and [uv](https://github.com/astral-sh/uv).

### Setup

```bash
git clone <repo-url>
cd spam_detector
uv sync
```

### Development tooling

```bash
uv sync --extra dev
```

Installs `ruff` and `pytest`. They sit in the `dev` extra, which the default `uv sync` does not pull in, so `uv run ruff check app/` fails until this command has been run.

### Teardown

```bash
rm -rf .venv models/
```

Deleting `models/` is irreversible from the repository: nothing under it is tracked in git. The trained checkpoints can only come back from a full notebook run, timed below. The vocabulary artefacts are the exception -- `mkdir -p models && uv run python scripts/export_vocab.py`, run from the repository root, rebuilds `vocab.pkl` and `tokenizer_config.pkl` identically (3,547 entries, `max_len` 32) from the tracked `data/processed/train.csv`.

## Usage

### Streamlit App

```bash
uv run streamlit run app/main.py
```

Four pages: Accueil (overview), Exploration (EDA), Resultats (model comparison), Demo (live prediction).

Launch the app through `app/main.py` only. Three of the four pages -- Accueil, Exploration and Demo -- import `app/text_cleaning.py` as a sibling module, which resolves because `streamlit run` puts the entry script's directory on `sys.path`. Loading one of those three through `streamlit.testing.v1.AppTest` raises `ModuleNotFoundError: No module named 'text_cleaning'` unless `app/` is added to `sys.path` first; Resultats imports no sibling module, builds its own `ROOT` from `__file__`, and renders under AppTest with no such patch. Running `python app/1_accueil.py` directly does resolve the import -- CPython puts the script's own directory on `sys.path` -- but it exits without rendering anything, there being no Streamlit runtime.

**A fresh clone cannot run the full app.** `models/` is excluded by `.gitignore` and no file under it is tracked, so a clone contains no trained artifacts: the Resultats page reports that no model results were found, and the Demo page reports a missing vocabulary and stops. Accueil and Exploration work out of the box because they read `data/`, which is tracked. Run the notebook end to end first to repopulate `models/`.

### Notebook

```bash
uv run jupyter notebook spam-detection_DL.ipynb
```

`spam-detection_DL.ipynb` is the deliverable: it runs the whole pipeline, from the raw CSV to the five trained models, and writes everything under `models/` and `plots/`. The other notebooks in `notebooks/` are earlier exploratory drafts, superseded by this one. `01_eda.ipynb` and `02_preprocessing.ipynb` carry their stored outputs; `03_model_simple.ipynb` and `04_transfer_learning.ipynb` were never executed -- zero executed cells and zero stored outputs -- so their prose states intent, not a recorded result.

Cost of a full run, as recorded rather than estimated:

- **3h07 wall clock on CPU** for the committed run, read from the notebook's own log timestamps (10:03:28 to 13:10:41).
- **2h43 of that is model training**, the sum of the five durations the notebook measures and writes to `models/*_results.json` (679 + 221 + 4060 + 2057 + 2764 s).
- The GAN generator's own training time **is never measured**. It can only be inferred from the surrounding log timestamps, at roughly 21 minutes, and should be treated as an estimate rather than a figure.

## Dataset

SMS Spam Collection (UCI), `data/input/spam.csv`.

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
