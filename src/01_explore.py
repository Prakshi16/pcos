"""Step 1: Exploratory data analysis on the raw PCOS dataset.

Reads: data/raw/PCOS_data_without_infertility.xlsx (sheet Full_new).
Writes: results/01_explore/ (column dump, missing-value report, class balance,
        summary statistics, correlation heatmap + high-correlation pairs,
        per-class distribution plots for key features).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config
from src.utils.plotting import new_fig, save_fig

OUT = config.RESULTS_DIR / "01_explore"
OUT.mkdir(parents=True, exist_ok=True)


def normalize_columns(df):
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip()) for c in df.columns]
    return df


def main():
    xl = pd.ExcelFile(config.RAW_XLSX)
    if config.RAW_SHEET not in xl.sheet_names:
        raise ValueError(
            f"Sheet '{config.RAW_SHEET}' not found. Available sheets: {xl.sheet_names}"
        )
    df = xl.parse(config.RAW_SHEET)
    df = normalize_columns(df)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from sheet '{config.RAW_SHEET}'.")

    # --- raw_columns.txt ---
    lines = []
    for c in df.columns:
        example = df[c].dropna().iloc[0] if df[c].notna().any() else None
        lines.append(f"{c!r}\tdtype={df[c].dtype}\texample={example!r}")
    (OUT / "raw_columns.txt").write_text("\n".join(lines))

    # --- unique_values_categorical.txt ---
    cat_lines = []
    for c in df.columns:
        n_unique = df[c].nunique(dropna=True)
        if n_unique <= 10:
            vals = df[c].value_counts(dropna=False).to_dict()
            cat_lines.append(f"{c!r}: {vals}")
    (OUT / "unique_values_categorical.txt").write_text("\n".join(cat_lines))
    print(f"Found {len(cat_lines)} columns with <=10 unique values (see unique_values_categorical.txt).")

    # --- missing_values.csv ---
    missing_rows = []
    for c in df.columns:
        raw_missing = df[c].isna().sum()
        coerced = pd.to_numeric(df[c], errors="coerce")
        non_numeric = coerced.isna().sum() - raw_missing
        missing_rows.append({
            "column": c,
            "missing_count": int(raw_missing),
            "non_numeric_count": int(max(non_numeric, 0)),
        })
    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(OUT / "missing_values.csv", index=False)

    # --- class_balance.png ---
    target_col = "PCOS (Y/N)"
    counts = df[target_col].value_counts(dropna=False)
    print("Class counts (PCOS Y/N):")
    print(counts)
    fig, ax = new_fig((5, 4))
    counts.sort_index().plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_xlabel("PCOS (Y/N)")
    ax.set_ylabel("Count")
    ax.set_title("Class balance")
    save_fig(fig, OUT / "class_balance.png")

    # --- summary_statistics.csv (numeric columns, split by class) ---
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_cols = [c for c in numeric_df.columns if numeric_df[c].notna().sum() > 0 and c != target_col]
    summary = numeric_df.groupby(numeric_df[target_col])[numeric_cols].describe().T
    summary.to_csv(OUT / "summary_statistics.csv")

    # --- correlation_heatmap.png + high_correlation_pairs.csv ---
    corr = numeric_df[numeric_cols + [target_col]].corr(method="pearson")
    fig, ax = new_fig((14, 12))
    import seaborn as sns
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True,
                cbar_kws={"shrink": 0.6}, xticklabels=True, yticklabels=True)
    ax.set_title("Correlation heatmap (Pearson, numeric features)")
    save_fig(fig, OUT / "correlation_heatmap.png")

    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > 0.8:
                pairs.append({"feature_1": cols[i], "feature_2": cols[j], "pearson_r": r})
    pairs_df = pd.DataFrame(pairs).sort_values("pearson_r", key=abs, ascending=False)
    pairs_df.to_csv(OUT / "high_correlation_pairs.csv", index=False)
    print(f"{len(pairs_df)} feature pairs with |r| > 0.8 (see high_correlation_pairs.csv).")

    # --- distributions_by_class/ ---
    dist_dir = OUT / "distributions_by_class"
    dist_dir.mkdir(exist_ok=True)
    key_features_source = {
        "bmi": "BMI",
        "cycle_length": "Cycle length(days)",
        "follicle_left": "Follicle No. (L)",
        "follicle_right": "Follicle No. (R)",
        "amh": "AMH(ng/mL)",
        "age": "Age (yrs)",
    }
    for canon, src_col in key_features_source.items():
        if src_col not in df.columns:
            continue
        vals = pd.to_numeric(df[src_col], errors="coerce")
        plot_df = pd.DataFrame({src_col: vals, target_col: df[target_col]}).dropna()
        fig, ax = new_fig((6, 4))
        for cls, color in [(0, "#4C72B0"), (1, "#C44E52")]:
            subset = plot_df.loc[plot_df[target_col] == cls, src_col]
            sns.kdeplot(subset, ax=ax, label=f"PCOS={cls}", fill=True, alpha=0.3, color=color)
        ax.set_title(f"{src_col} distribution by class")
        ax.legend()
        save_fig(fig, dist_dir / f"{canon}_by_class.png")

    # lh_fsh_ratio needs to be derived first
    if "LH(mIU/mL)" in df.columns and "FSH(mIU/mL)" in df.columns:
        lh = pd.to_numeric(df["LH(mIU/mL)"], errors="coerce")
        fsh = pd.to_numeric(df["FSH(mIU/mL)"], errors="coerce")
        ratio = lh / fsh.replace(0, np.nan)
        plot_df = pd.DataFrame({"lh_fsh_ratio": ratio, target_col: df[target_col]}).dropna()
        fig, ax = new_fig((6, 4))
        for cls, color in [(0, "#4C72B0"), (1, "#C44E52")]:
            subset = plot_df.loc[plot_df[target_col] == cls, "lh_fsh_ratio"]
            sns.kdeplot(subset, ax=ax, label=f"PCOS={cls}", fill=True, alpha=0.3, color=color)
        ax.set_title("lh_fsh_ratio distribution by class")
        ax.legend()
        save_fig(fig, dist_dir / "lh_fsh_ratio_by_class.png")

    print("Step 1 (01_explore.py) complete. Outputs written to results/01_explore/.")
    print("Please verify Cycle(R/I) coding and column names against Section 3.2 (task M5).")


if __name__ == "__main__":
    main()
