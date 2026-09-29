"""Step 2: Clean the raw dataset, derive features, split train/test, save input schema.

Reads: data/raw/PCOS_data_without_infertility.xlsx (sheet Full_new).
Writes: data/processed/clean.csv, data/processed/train.csv, data/processed/test.csv,
        models/input_schema.json, results/01_explore/implausible_values.csv.

Implements Section 3.3 exactly. No imputation happens here — imputation is
performed inside model pipelines (04-07) to avoid data leakage.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def normalize_columns(df):
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip()) for c in df.columns]
    return df


def load_raw():
    xl = pd.ExcelFile(config.RAW_XLSX)
    if config.RAW_SHEET not in xl.sheet_names:
        raise ValueError(f"Sheet '{config.RAW_SHEET}' not found. Available: {xl.sheet_names}")
    df = xl.parse(config.RAW_SHEET)
    df = normalize_columns(df)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    return df


def check_columns(df):
    expected_source_cols = (
        set(config.COLUMN_MAPPING.keys())
        | set(config.DROP_COLUMNS.keys())
    )
    missing = sorted(c for c in expected_source_cols if c not in df.columns)
    if missing:
        raise ValueError(
            "Expected source columns missing from raw file.\n"
            f"Missing: {missing}\n"
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def main():
    df = load_raw()
    check_columns(df)
    n_raw = len(df)
    print(f"Loaded {n_raw} raw rows.")

    # Drop rows missing the target BEFORE any other processing.
    target_source = "PCOS (Y/N)"
    df = df[df[target_source].notna()].copy()
    print(f"Dropped {n_raw - len(df)} rows with missing target. {len(df)} rows remain.")

    # Drop explicitly excluded columns and helper-only columns are kept for now
    # (needed to recompute bmi / waist_hip_ratio) then dropped at the end.
    df = df.drop(columns=list(config.DROP_COLUMNS.keys()), errors="ignore")

    # Rename mapped columns to canonical names.
    df = df.rename(columns=config.COLUMN_MAPPING)

    # 1. Coerce every non-target column to numeric, logging NaNs introduced.
    non_target_cols = [c for c in df.columns if c != "pcos"]
    coercion_log = []
    for c in non_target_cols:
        before_na = df[c].isna().sum()
        coerced = pd.to_numeric(df[c], errors="coerce")
        after_na = coerced.isna().sum()
        introduced = after_na - before_na
        if introduced > 0:
            coercion_log.append({"column": c, "new_nans_from_coercion": int(introduced)})
        df[c] = coerced
    coercion_df = pd.DataFrame(coercion_log)
    print(f"Numeric coercion introduced NaNs in {len(coercion_df)} columns.")

    # 2. Binary Y/N source columns -> 0/1. Cycle(R/I): 2->0, 4->1, else NaN.
    binary_canon_from_source = [config.COLUMN_MAPPING[c] for c in config.BINARY_SOURCE_COLUMNS]
    for c in binary_canon_from_source:
        # Source was already Y/N text; after pd.to_numeric it's all-NaN unless it
        # was numeric to begin with. Re-derive from the raw (pre-coercion) frame.
        pass

    # Re-load raw Y/N columns before numeric coercion clobbered them.
    raw_for_binary = load_raw().rename(columns=config.COLUMN_MAPPING)
    raw_for_binary = raw_for_binary.loc[df.index.intersection(raw_for_binary.index)]
    for src_col in config.BINARY_SOURCE_COLUMNS:
        canon = config.COLUMN_MAPPING[src_col]
        raw_vals = raw_for_binary[canon]
        mapped = raw_vals.map({1: 1, 0: 0, "Y": 1, "N": 0, "y": 1, "n": 0})
        # Values already numeric 0/1 pass through map's numeric keys; anything else -> NaN
        mapped = mapped.where(mapped.isin([0, 1]), np.nan)
        df[canon] = mapped

    df["cycle_irregular"] = raw_for_binary["cycle_irregular"].map({2: 0, 4: 1})
    df["cycle_irregular"] = df["cycle_irregular"].where(df["cycle_irregular"].isin([0, 1]), np.nan)

    # 3. Implausible values -> NaN, logged for review.
    implausible_rows = []
    for feat, (lo, hi) in config.PLAUSIBLE_RANGES.items():
        if feat not in df.columns:
            continue
        mask = df[feat].notna() & ((df[feat] < lo) | (df[feat] > hi))
        for idx, val in df.loc[mask, feat].items():
            implausible_rows.append({"row_index": idx, "feature": feat, "value": val,
                                      "plausible_min": lo, "plausible_max": hi})
        df.loc[mask, feat] = np.nan
    implausible_df = pd.DataFrame(implausible_rows)
    out_explore = config.RESULTS_DIR / "01_explore"
    out_explore.mkdir(parents=True, exist_ok=True)
    implausible_df.to_csv(out_explore / "implausible_values.csv", index=False)
    print(f"{len(implausible_df)} implausible values replaced with NaN (see results/01_explore/implausible_values.csv).")

    # 4. Derived features.
    valid_hw = df["height"].notna() & df["weight"].notna() & (df["height"] > 0)
    recomputed_bmi = df["weight"] / ((df["height"] / 100) ** 2)
    df["bmi"] = np.where(valid_hw, recomputed_bmi, df["bmi"])

    valid_wh = df["waist"].notna() & df["hip"].notna() & (df["hip"] > 0)
    df["waist_hip_ratio"] = np.where(valid_wh, df["waist"] / df["hip"], np.nan)

    fsh_safe = df["fsh"].replace(0, np.nan)
    df["lh_fsh_ratio"] = df["lh"] / fsh_safe

    # Re-apply plausibility check to recomputed bmi / waist_hip_ratio.
    for feat in ["bmi", "waist_hip_ratio"]:
        lo, hi = config.PLAUSIBLE_RANGES[feat]
        mask = df[feat].notna() & ((df[feat] < lo) | (df[feat] > hi))
        df.loc[mask, feat] = np.nan

    # Drop helper-only columns.
    df = df.drop(columns=config.HELPER_COLUMNS, errors="ignore")

    # Keep only canonical columns actually used downstream.
    keep_cols = ["pcos"] + config.TIER_FEATURES["tier3_full"] + ["lh_fsh_ratio"]
    keep_cols = [c for c in dict.fromkeys(keep_cols) if c in df.columns]
    df = df[keep_cols]

    df.to_csv(config.PROCESSED_DIR / "clean.csv", index=False)
    print(f"Saved clean.csv with {len(df)} rows, {len(df.columns)} columns.")

    # 7. Train/test split, stratified by pcos.
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=config.SEED, stratify=df["pcos"]
    )
    train_df.to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(config.PROCESSED_DIR / "test.csv", index=False)
    print("Train class counts:")
    print(train_df["pcos"].value_counts())
    print("Test class counts:")
    print(test_df["pcos"].value_counts())

    # 8. Input schema for the app (computed on train only, no imputation).
    schema = {}
    all_features = config.TIER_FEATURES["tier3_full"] + ["lh_fsh_ratio"]
    all_features = list(dict.fromkeys(all_features))
    for feat in all_features:
        if feat not in train_df.columns:
            continue
        is_binary = feat in config.BINARY_FEATURES
        tier = next((t for t in config.TIERS if feat in config.TIER_FEATURES[t]), "tier2_lab")
        col = train_df[feat].dropna()
        entry = {
            "canonical_name": feat,
            "label": config.FEATURE_LABELS.get(feat, feat),
            "type": "binary" if is_binary else "numeric",
            "tier": tier,
        }
        if not is_binary and len(col) > 0:
            entry["p1"] = float(np.percentile(col, 1))
            entry["p50"] = float(np.percentile(col, 50))
            entry["p99"] = float(np.percentile(col, 99))
        schema[feat] = entry

    from src.utils.io import save_json
    save_json(schema, config.MODELS_DIR / "input_schema.json")
    print("Saved models/input_schema.json.")
    print("Step 2 (02_clean_and_split.py) complete.")


if __name__ == "__main__":
    main()
