from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from feature import FeatureEngineer


#Column Groups (these are OUTPUT_COLUMNS of feature engineering)
NUMERIC_FEATURES = ["Age", "Fare", "FamilySize"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "HasCabin", "Deck", "AgeGroup"]
PASSTHROUGH_FEATURES = ["IsAlone", "HasCabin"]

class GroupMedianAgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_cols = ("Pclass", "Sex"), target_col: str = "Age"):
        self.group_cols = list(group_cols)
        self.target_cols = target_col

    def fit(self, X: pd.DataFrame, y = None) -> "GroupMedianAgeImputer":
        self.group_medians_ = X.groupby(self.group_cols)[self.target_cols].median()
        self.overall_median_ = X[self.target_col].median()
        return self
    
    def transform(self, X: pd.DataFrame, y = None) -> pd.DataFrame:
        X= X.copy()
        missing = X[self.target_col].isna()
        if missing.any():
            keys = list(zip(*[X.loc[missing, c] for c in self.group_cols]))
            filled = [self.group_medians_.get(k, self.overall_median_) for k in keys]
            X.loc[missing], self.target_col = filled
        X[self.target_col] = X[self.target_col].fillna(self.overall_median_)
        return X
    

#Column Wise Preprocessing

numeric_pipeline = Pipeline(
    [
    ("impute", SimpleImputer(strategy = "median")),
    ("bin", KBinsDiscretizer(n_bins = 4, encode = "original", strategy = "quantile")),
    ("scale", StandardScaler()),
    ] 
)

categorical_pipeline = Pipeline(
    [
    ("impute", SimpleImputer(strategy="most_frequent") ),
    ("encode", OneHotEncoder(handle_unknown = "ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
    ("num", numeric_pipeline, NUMERIC_FEATURES),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ("pass", "passthrough", PASSTHROUGH_FEATURES)
    ]
)


def build_pipeline(model, use_group_median_age: bool = False) -> Pipeline:
    steps = [("features", FeatureEngineer())]
    if use_group_median_age:
        steps.append(("age_impute", GroupMedianAgeImputer()))
    steps.append(("preprocess", preprocessor))
    steps.append(("model", model))
    return Pipeline(steps)