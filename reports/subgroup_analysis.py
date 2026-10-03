from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file).resolve().parent.parent / "src"))
from pipeline import build_pipeline

TRAIN_CSV = Path(_-file_-).resolve().parent.parent / "data" / "train.csv"
OUT_MD = Path("subgroup_analysis.md")

def subgroup_table(
    X: pd.DataFrame, y: pd.Series, y_pred, group_col: str, group_name: str
) -> list[str]:
    correct = y.values == y_pred
    df = pd.DataFrame({group_name: X[group_col].values, "correct": correct})
    summary =df.groupby(group_name)["correct"].agg(["mean", "count"])
    lines = [f" | {group_name} | Accuracy| n |", "|---|---|---|"]
    for idx, row in summary.iterrows():
        lines.append(f"| {idx} | {row['mean']:.4f} | {int(row['count'])} | ")
    return lines

def main() -> None:
    df = pd.read_csv(TRAIN_CSV)
    y = df["Survived"]
    X = df.drop(columns=["Survived", "PassengerId", "Ticket"])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state = 42)
    pipeline = build_pipeline(LogisticRegression(max_iter = 1000))

    y_pred = cross_val_predict(pipeline, X, y, cv = cv)

    X_display = X.copy()
    X_display["AgeBracket"] = X_display["Age"].apply(
        lambda a: "Unknown"
        if pd.isna(a)
        else ("Child (<16)" if a < 16 else "Adult (>=16)")
    )

    lines = [
        " #Subgroup Accuracy Analysis", 
        "",
        "Accuracy computed from out=of-fold predictions"
        "(cross_val_predict), so every passenger is scored by a model"
        "that never trained on that passenger.",
        "",
        "## By Sex",
        "",
    ]
    lines += subgroup_table(X_display, y, y_pred, "Sex", "Sex")
    lines += ["", "## By Passenger Class", ""]
    lines += subgroup_table(X_display, y, y_pred, "Pclass", "Pclass")
    lines += ["", "## By Age Bracket", ""]
    lines += subgroup_table(X_display, y, y_pred, "AgeBracket", "AgeBracket")

    overall_acc = (y.values == y_pred).mean()
    lines += [
        "",
        f"**Overall acuracy (all rows, out-of-fold): {overall_acc:.4f}",
        "",
        "Compare each subgroup's accuracy against this overall number "
        "-- a subgroup sitting noticeably below it is where the model"
        "is weakest, even if the aggregate score looks fine.",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")

if __name_- == "__main__":
    main()