"""Tests for `clean_text` and its parity with the notebook version."""

import ast
import json
import re
from pathlib import Path

import pandas as pd
import pytest
from text_cleaning import clean_text

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "spam-detection_DL.ipynb"
CLEAN_TEXT_CELL = 13
RAW_CSV = ROOT / "data" / "input" / "spam.csv"

CASES = [
    ("Visit HTTP://Example.com/x now", "visit URL now"),
    ("go www.example.co.uk", "go URL"),
    ("call 555-123-4567", "call PHONE"),
    ("txt 87121 to win", "txt PHONE to win"),
    ("  A \n b  ", "a b"),
    ("1234", "1234"),
    (None, ""),
]


@pytest.mark.parametrize(("raw", "expected"), CASES)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


@pytest.fixture(scope="module")
def notebook_clean_text():
    """Compile `clean_text` from the notebook cell, without running the notebook."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    tree = ast.parse("".join(cells[CLEAN_TEXT_CELL]["source"]))
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "clean_text"
    )
    module = ast.Module(body=[func], type_ignores=[])
    namespace = {"re": re}
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace["clean_text"]


@pytest.mark.parametrize("raw", [raw for raw, _ in CASES])
def test_notebook_parity_on_cases(notebook_clean_text, raw):
    assert notebook_clean_text(raw) == clean_text(raw)


def test_notebook_parity_on_corpus(notebook_clean_text):
    df = pd.read_csv(RAW_CSV, encoding="latin-1", usecols=[0, 1])
    texts = df.iloc[:300, 1].tolist()
    assert [notebook_clean_text(t) for t in texts] == [clean_text(t) for t in texts]
