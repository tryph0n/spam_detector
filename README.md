# SMS Spam Detector

Deep learning SMS spam classifier comparing five neural architectures with a Streamlit web app for live predictions.

## Results

Test set: 772 samples (678 ham, 94 spam).

| Model | Params | Accuracy | F1 | Training time |
|-------|--------|----------|------|---------------|
| BiLSTM | 553K | 0.959 | 0.845 | 679s |
| TextCNN | 608K | 0.977 | 0.902 | 221s |
| RoBERTa | 125M | 0.992 | 0.968 | 4060s |
| DistilBERT | 67M | 0.991 | 0.962 | 2057s |
| BiLSTM GAN-Augmented | 567K | 0.972 | 0.879 | 2764s |

RoBERTa achieves the best F1 (0.968). DistilBERT is nearly equivalent (0.962) at half the training time.

## Installation

Prerequisites: Python 3.11+ and [uv](https://github.com/astral-sh/uv).

### Setup

```bash
git clone <repo-url>
cd spam_detector
uv sync
```

### Teardown

```bash
rm -rf .venv models/
```

## Usage

### Streamlit App

```bash
uv run streamlit run app/main.py
```

Four pages: Accueil (overview), Exploration (EDA), Resultats (model comparison), Demo (live prediction).

### Notebook

```bash
uv run jupyter notebook final_deliverable.ipynb
```

## Dataset

SMS Spam Collection -- 5,572 messages (87% ham / 13% spam).

## Tech Stack

| Category | Technology |
|----------|------------|
| Deep Learning | PyTorch, HuggingFace Transformers |
| NLP | scikit-learn |
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web App | Streamlit |
| Package Manager | uv |
