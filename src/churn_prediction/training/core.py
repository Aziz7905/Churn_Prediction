from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from churn_prediction.config import TARGET_COLUMN_CANDIDATES


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def resolve_target_column(columns: list[str]) -> str:
    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(
        "Could not find a churn target column. Expected one of: "
        + ", ".join(TARGET_COLUMN_CANDIDATES)
    )


def prepare_data(train_path: str | Path | None, test_path: str | Path, scaler: StandardScaler | None = None):
    """
    Prepare train and test data with matching one-hot columns and feature scaling.

    If `train_path` is None, only the test set is processed with the provided scaler.
    """
    test_path = _to_path(test_path)

    if train_path is not None:
        train_path = _to_path(train_path)
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        df_train = pd.get_dummies(df_train, drop_first=True)
        df_test = pd.get_dummies(df_test, drop_first=True)
        df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

        target_column = resolve_target_column(df_train.columns.tolist())
        numeric_columns = [
            column
            for column in df_train.select_dtypes(include=[np.number]).columns.tolist()
            if column != target_column
        ]

        if scaler is None:
            scaler = StandardScaler()
            df_train[numeric_columns] = scaler.fit_transform(df_train[numeric_columns])
        else:
            df_train[numeric_columns] = scaler.transform(df_train[numeric_columns])

        df_test[numeric_columns] = scaler.transform(df_test[numeric_columns])
        return df_train, df_test, scaler

    df_test = pd.read_csv(test_path)
    df_test = pd.get_dummies(df_test, drop_first=True)
    if scaler is not None:
        target_column = resolve_target_column(df_test.columns.tolist())
        numeric_columns = [
            column
            for column in df_test.select_dtypes(include=[np.number]).columns.tolist()
            if column != target_column
        ]
        df_test[numeric_columns] = scaler.transform(df_test[numeric_columns])
    return None, df_test, scaler


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = predictions

    acc = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, scores)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    return acc, auc, report, matrix


def save_model(model, scaler, filename: str | Path = "model.pkl") -> None:
    filename = _to_path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, scaler), filename)


def load_model(filename: str | Path = "model.pkl"):
    return joblib.load(_to_path(filename))
