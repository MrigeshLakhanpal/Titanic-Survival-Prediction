from __future__ import annotations

import argparse

import joblib
import pandas as pd


def main(model_path: str, test_path: str, out_path: str) -> None:
    pipeline = joblib.load(model_path)

    test_df = pd.read_csv(test_path)
    passenger_ids = test_df["PassengerId"]

    # Same column drop as train.py's X -- the pipeline was fit expecting
    # exactly this column set (everything except Survived, PassengerId,
    # Ticket). test.csv has no Survived column at all, which is correct:
    # that's the thing we're predicting. Any missing values in test.csv
    # (e.g. the one missing Fare) are handled by the imputer's medians
    # already learned from the training set -- no special-casing needed.
    X_test = test_df.drop(columns=["PassengerId", "Ticket"])

    predictions = pipeline.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions.astype(int),
    })

    assert len(submission) == 418, (
        f"Expected 418 rows (Kaggle's fixed test set size), got "
        f"{len(submission)} -- check --test points at the real test.csv."
    )
    assert list(submission.columns) == ["PassengerId", "Survived"], (
        "Kaggle rejects submissions with extra or reordered columns."
    )

    submission.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(submission)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="../model.joblib")
    parser.add_argument("--test", default="../data/test.csv")
    parser.add_argument("--out", default="../submission.csv")
    args = parser.parse_args()
    main(args.model, args.test, args.out)
