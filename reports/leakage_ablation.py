from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

sys.path.insert(0, str(Path(__file__).resolved().parent.parent / "src"))
from feature import FeatureEngineer
from pipeline import build_pipeline

TRAIN_CSV = Path(__file__).resolved().parent.parent / "data" / "train.csv"
OUT_MD = Path("leakage_ablation.md")

def build_leaky_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Every Statistics below is computed on the full dataframe passed in, before any train/ validation split exists
    This is the leak the ablation measures."""
    X = FeatureEngineer().fit_transform(df)

    #Leaky: a single global median, not fit per fold
    X["Age"] = X["Age"].fillna(X["Age"].median())
    X["Fare"] = X["Fare"].fillna(X["Fare"].median())

    #Leaky: quantiles bin edges computed on the full dataset, not per fold
    X["FareBin"] = pd.qcut(X["Fare"], q=4, labels=False, duplicates="drop")
    X = X.drop(columns=["Fare"])

    # Not a statistical leak by itself
    X = pd.get_dummies(
        X, columns = ["PClass", "Sex", "Embarked", "Title", "Deck","AgeGroup"]
    )
    return X


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)
    y = df["Survived"]
    X_raw = df.drop(columns=["Survived", "PassengerId", "Ticket"])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #Leaky preprocess globally, Then cross_validatea bare model
    X_leaky = build_leaky_matrix(X_raw)
    leaky = cross_validate(
        LogisticRegression(max_iter=1000), X_leaky, y, cv=cv,
        scoring = ["accuracy", "roc_auc"],
    )

    #Safe: whole piprline refits inside each fold
    safe = cross_validate(
        build_pipeline(LogisticRegression(max_iter=1000)), X_raw, y, cv=cv,
        scoring = ["accuracy", "roc-auc"]
    )

    leaky_acc, leaky_auc = leaky["test_accuracy"], leaky["test_roc_auc"]
    safe_acc, safe_auc = safe["test_accuracy"], safe["test_roc_auc"]
    gap_acc = leaky_acc.mean() - safe_acc.mean()
    gap_auc = leaky_auc.mean() - safe_auc.mean()

    lines = [
        "# Leakage Ablation",
        "",
        "Same model (Logistic Regression), same 5-fold StratifiedKFold "
        "split, two different preprocessing orders.",
        "",
        "| Version | Accuracy | ROC-AUC |",
        "|---|---|---|",
        f"| Leaky (global median + global quantile bins, fit before split) "
        f"| {leaky_acc.mean():.4f} \u00b1 {leaky_acc.std():.4f} "
        f"| {leaky_auc.mean():.4f} \u00b1 {leaky_auc.std():.4f} |",
        f"| Safe (this project's Pipeline, refit per fold) "
        f"| {safe_acc.mean():.4f} \u00b1 {safe_acc.std():.4f} "
        f"| {safe_auc.mean():.4f} \u00b1 {safe_auc.std():.4f} |",
        "",
        f"**Gap:** {gap_acc:+.4f} accuracy, {gap_auc:+.4f} ROC-AUC "
        "(leaky minus safe).",
        "",
        "A positive gap means the leaky version reports a better score "
        "than the model would actually achieve on genuinely unseen "
        "data \u2014 the inflation comes entirely from letting validation-"
        "fold rows influence the median and quantile-bin statistics "
        "used to preprocess the training rows in the same fold.",
        "",
        "Note: at this dataset size (891 rows, 5 folds) the numeric "
        "gap can be small or noisy \u2014 the point of this experiment "
        "is demonstrating that the mechanism exists and reporting the "
        "measured gap honestly, not manufacturing a large effect.",
    ]

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")
    print(f"Gap -- Accuracy: {gap_acc:+.4f}  ROC-AUC: {gap_auc:+.4f}")

if __name__ == "__main__":
    main()