from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from feature import FeatureEngineer
from pipeline import build_pipeline, preprocessor

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "train.csv"

@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

def test_feature_engineer_output_columns(raw_df):
    out = FeatureEngineer().fit_transform(raw_df.head(20))
    assert list(out.columns) == FeatureEngineer.OUTPUT_COLUMNS
    assert len(out) == 20

def test_title_extraction_is_clean(raw_df):
    out = FeatureEngineer().fit_transform(raw_df)
    assert out["Title"].nunique() <= 6

def test_no_nans_after_preprocessing(raw_df):
    engineered = FeatureEngineer().fit_transform(raw_df)
    encoded = preprocessor.fit_transform(engineered)
    encoded_dense = encoded.toarray() if hasattr(encoded, "toarray") else encoded
    assert np.isnan(encoded_dense).sum() == 0

def test_imputer_statistics_differ_across_subsets(raw_df):
    fe = FeatureEngineer()
    half1 = fe.fit_transform(raw_df.iloc[:400])
    half2 = fe.fit_transform(raw_df.iloc[400:])

    p1 = clone(preprocessor).fit(half1)
    p2 = clone(preprocessor).fit(half2)

    age_median_1 = p1.named_transformers_["num"].named_steps["impute"].statistics_[0]
    age_median_2 = p2.named_transformers_["num"].named_steps["impute"].statistics_[0]

    assert age_median_1 != age_median_2, (
        "Age imputer learned the same value from two different subsets "
        "-- this would only happen if the statistic were computed "
        "globally rather than per-fold."
    )

def test_pipeline_fits_and_predicts_binary(raw_df):
    X = raw_df.drop(columns=["Survived", "PassengerId", "Ticket"])
    y = raw_df["Survived"]
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert len(preds) == len(X)