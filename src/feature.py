"""
features.py - Row-wise feature engineering for the titanic dataset.

Every function and transformer in this file operates on ONE row at a time.
Nothing here computes a statistic across rows, so this step is leak free by construction: 
applying it before or after the train/validation split produces identical output.
"""

from __future__ import annotations

import re 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Title Extraction
TITLE_SYNONYMS = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}

RARE_TITLES = {
    "Lady", "the Countess","Countess", "Capt", "Col", "Don","Dr", "Major"
    ,"Rev", "Sir", "Jonkheer", "Dona"
}

def extract_title(name: str) -> str:
    match = re.search(r",\s*([^.]*)\.", str(name))
    title = match.group(1).strip() if match else "Unknown"
    title = TITLE_SYNONYMS.get(title, title)
    return "Rare" if title in RARE_TITLES else title


#Fixed - edge binning

AGE_BIN_EDGES = [0, 12, 18, 35, 60, 80]
AGE_BIN_LABELS = ["Child", "Teen", "Young Adult", "Adult", "Senior"]

def age_group(age: float) -> str:
    if pd.isna(age):
        return "Unknown"
    label = pd.cut([age], bins = AGE_BIN_EDGES, labels = AGE_BIN_LABELS)[0]
    return str(label) if pd.notna(label) else "Unknown"


#Row wise Feature engineer
class FeatureEngineer(BaseEstimator, TransformerMixin):
    OUTPUT_COLUMNS = [
        "Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "FamilySize",
        "IsAlone", "HasCabin", "Deck", "AgeGroup",
    ]

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self
    def transform(self, X:pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        X["Title"] = X["Name"].apply(extract_title)
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        X["HasCabin"] = X["Cabin"].notna().astype(int)
        X["Deck"] = X["Cabin"].astype(str).str[0].where(X["Cabin"].notna(), "U")
        X["AgeGroup"] = X["Age"].apply(age_group)
        return X[self.OUTPUT_COLUMNS]