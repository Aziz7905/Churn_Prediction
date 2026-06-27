from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MLFLOW_DIR = ARTIFACTS_DIR / "mlruns"

LEGACY_MODEL_PATH = PROJECT_ROOT / "model.pkl"
LEGACY_PREPARED_DATA_PATH = PROJECT_ROOT / "prepared_data.pkl"
LEGACY_PREPARED_TEST_PATH = PROJECT_ROOT / "prepared_data_test.pkl"
LEGACY_CLASSIFICATION_REPORT_PATH = PROJECT_ROOT / "classification_report.txt"
LEGACY_EVALUATION_RESULTS_PATH = PROJECT_ROOT / "evaluation_results.txt"

DEFAULT_MODEL_PATH = MODELS_DIR / "churn_model.pkl"
DEFAULT_PREPARED_DATA_PATH = PROCESSED_DIR / "prepared_data.pkl"
DEFAULT_PREPARED_TEST_PATH = PROCESSED_DIR / "prepared_data_test.pkl"
DEFAULT_CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"
DEFAULT_EVALUATION_RESULTS_PATH = REPORTS_DIR / "evaluation_results.txt"

TRAIN_DATA_FILENAME = "churn-bigml-80.csv"
TEST_DATA_FILENAME = "churn-bigml-20.csv"
TARGET_COLUMN_CANDIDATES = ("Churn", "Churn_Yes")


def ensure_directories() -> None:
    for path in (DATA_DIR, RAW_DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, MLFLOW_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def mlflow_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", f"file://{MLFLOW_DIR.resolve()}")


def churn_api_url() -> str:
    return os.getenv("CHURN_API_URL", "http://localhost:8000/predict/")


def elasticsearch_url() -> str:
    return os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

