"""Load/save helpers shared across scripts (csv, json, joblib, markdown tables)."""
import json
import joblib
import pandas as pd
from pathlib import Path


def save_csv_and_md(df: pd.DataFrame, path_no_ext: Path, index=False):
    """Write a DataFrame to both <path>.csv and <path>.md (markdown table)."""
    path_no_ext = Path(path_no_ext)
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_no_ext.with_suffix(".csv"), index=index)
    with open(path_no_ext.with_suffix(".md"), "w") as f:
        f.write(df.to_markdown(index=index))


def save_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_model(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_model(path: Path):
    return joblib.load(path)
