from __future__ import annotations
from pathlib import Path

import pandas as pd

RAW_CSV = Path("baseline_vs_model_raw.csv")
OUT_MD = Path("baseline_vs_model.md")

def format_row(row: pd.Series) -> str:
    auc = (
        f"{row['roc_auc_mean']:.4f} \u00b1 {row['roc_auc_std']:.4f}"
        if pd.notna(row["roc_auc_mean"])
        else "\u2014 (not a scored classifier)"
    )
    return (
        f"| {row['model']}"
        f"| {row['accuracy_mean']:.4f} \u00b1 {row['accuracy_std']:.4f}"
        f"| {auc} |"
    )

def main() -> None:
    df = pd.read_csv(RAW_CSV)
    best_row = df.loc[df["roc_auc_mean"].idxmax()]

    lines = [
        "",
        "5-fold stratified cross-validation, mean \u00b1 standard deviation",
        "across folds.",
        "",
        "| Model | Accuracy | ROC-AUC |",
        "|-------|----------|---------|",
    ]
    lines += [format_row(row) for _, row in df.iterrows()]
    lines += [
        "# Baseline vs Model Comparison",
        "",
        f"Best model by ROC-AUC: {best_row['model']}",
        f"ROC-AUC = {best_row['roc_auc_mean']:.4f} \u00b1 {best_row['roc_auc_std']:.4f}",
        "",
        "Any model scoring below the sex-only rule's accuracy has not "
        "learned anything beyond what a single column already reveals."
    ]

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")

if __name__ == "__main__":
    main()
    