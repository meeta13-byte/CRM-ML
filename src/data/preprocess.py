"""Sklearn preprocessing shared by both stages and by clustering."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import RANDOM_STATE, TEST_SIZE


def split_data(X, y, max_rows=None):
    """Stratified hold-out. Scaling is fitted inside the pipeline, never here."""
    if max_rows and len(X) > max_rows:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_rows, stratify=y, random_state=RANDOM_STATE
        )
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def build_preprocessor(numeric, categorical):
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


def build_pipeline(numeric, categorical, estimator):
    return Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),
        ("model", estimator),
    ])


def column_types(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical
