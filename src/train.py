from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold , cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pipeline import build_pipeline

def sex_only_baseline(X: pd.DataFrame, y: pd.Series, cv) -> tuple[float, float]:
    accs = []
    for _, test_idx in cv.split(X, y):
        pred = (X.iloc[test_idx]["Sex"] == "female").astype(int)
        accs.append((pred.values == y.iloc[test_idx].values).mean())
    accs = np.array(accs)
    print(f"{'Sex Only rule':28s} Accuracy = {accs.mean():.4f} \u00b1 {accs.std():.4f}")
    return accs.mean(), accs.std()

def evaluate(pipeline, X: pd.DataFrame, y: pd.Series, cv, name: str) -> dict:
    scores = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["accuracy", "roc_auc"],
        return_train_score=False,
    )
    acc, auc = scores["test_accuracy"], scores["test_roc_auc"]
    print(
        f"{name:28s} Accuracy = {acc.mean():.4f} \u00b1 {acc.std():.4f} "
        f"ROC-AUC = {auc.mean():.4f} \u00b1 {auc.std():.4f}"
    )
    return {
        "model": name,
        "accuracy_mean": acc.mean(), "accuracy_std": acc.std(),
        "roc_auc_mean": auc.mean(), "roc_auc_std": auc.std(),
    }

def main(train_path: str, model_out: str) -> None:
    df = pd.read_csv(train_path)
    y = df["Survived"]
    X = df.drop(columns=["Survived", "PassengerId", "Ticket"])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Baselines")
    print("-"*50)
    evaluate(
        build_pipeline(DummyClassifier(strategy="most_frequent")),
        X, y, cv, "Majority-class baseline",
    )
    sex_only_baseline(X, y, cv)

    print("\nModels")
    print("-"*50)
    candidates={
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forests": RandomForestClassifier(
            n_estimators=400, max_depth=6, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
        "Support Vector Machine": CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
        "Naive Bayes": GaussianNB(),
    }

    results=[]
    best_name, best_score, best_model = None, -1.0, None
    for name, model in candidates.items():
        result = evaluate(build_pipeline(model), X, y, cv, name)
        results.append(result)
        if result["roc_auc_mean"] > best_score:
            best_name, best_score, best_model = name, result["roc_auc_mean"], model

    print(f"\nBest model by ROC AUC: {best_name} ({best_score:.4f})")
    final_pipeline = build_pipeline(best_model)
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, model_out)
    print(f"Saved final pipeline to {model_out}")

    pd.DataFrame(results).to_csv("../reports/baseline_vs_model_raw.csv", index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_path", default="../data/train.csv")
    parser.add_argument("--out", default="../model.joblib")
    args = parser.parse_args()
    main(args.train_path, args.out)