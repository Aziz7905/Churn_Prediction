from __future__ import annotations

import logging
import joblib
import mlflow
import pandas as pd
from elasticsearch import Elasticsearch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from churn_prediction.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PREPARED_DATA_PATH,
    LEGACY_MODEL_PATH,
    LEGACY_PREPARED_DATA_PATH,
    elasticsearch_url,
    mlflow_tracking_uri,
    resolve_path,
)
from churn_prediction.training.core import resolve_target_column

logger = logging.getLogger("churn_api")
logger.setLevel(logging.INFO)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using a trained Decision Tree model",
    version="1.1.0",
    contact={"name": "Aziz Allah Barkaoui", "email": "aziz@example.com"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri(mlflow_tracking_uri())


def _load_bundle():
    model_path = resolve_path(DEFAULT_MODEL_PATH, LEGACY_MODEL_PATH)
    prepared_path = resolve_path(DEFAULT_PREPARED_DATA_PATH, LEGACY_PREPARED_DATA_PATH)
    if not model_path.exists() or not prepared_path.exists():
        raise RuntimeError(
            "Model artifacts are missing. Run `python run_training.py --mode train` first "
            "or provide existing model.pkl and prepared_data.pkl files."
        )
    model, scaler = joblib.load(model_path)
    prepared_data = pd.read_pickle(prepared_path)
    target_column = resolve_target_column(prepared_data.columns.tolist())
    expected_features = prepared_data.drop(columns=[target_column], errors="ignore").columns.tolist()
    return model, scaler, expected_features


MODEL = None
SCALER = None
EXPECTED_FEATURES = None


def _elasticsearch_client():
    try:
        return Elasticsearch(elasticsearch_url())
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Elasticsearch is unavailable: %s", exc)
        return None


ES_CLIENT = _elasticsearch_client()


@app.on_event("startup")
def load_artifacts() -> None:
    global MODEL, SCALER, EXPECTED_FEATURES

    try:
        MODEL, SCALER, EXPECTED_FEATURES = _load_bundle()
    except Exception as exc:  # pragma: no cover - startup fallback
        logger.warning("Model artifacts could not be loaded at startup: %s", exc)


def log_mlflow_to_es(run_id: str, metric_name: str, value: float) -> None:
    if ES_CLIENT is None:
        return

    try:
        ES_CLIENT.index(index="mlflow-metrics", document={"run_id": run_id, "metric_name": metric_name, "value": value})
    except Exception as exc:  # pragma: no cover - optional integration
        logger.warning("Failed to log metric to Elasticsearch: %s", exc)


class CustomerInput(BaseModel):
    State: str = Field(..., alias="State", description="US state abbreviation", example="LA")
    Account_length: int = Field(..., alias="Account length", ge=0, description="Months with the company", example=117)
    Area_code: int = Field(..., alias="Area code", description="Area code number", example=408)
    International_plan: str = Field(
        ..., alias="International plan", description="Whether the customer has an international plan", pattern="^(yes|no)$"
    )
    Voice_mail_plan: str = Field(..., alias="Voice mail plan", description="Whether the customer has a voice mail plan", pattern="^(yes|no)$")
    Number_vmail_messages: int = Field(..., alias="Number vmail messages", ge=0, description="Voice mail message count")
    Total_day_minutes: float = Field(..., alias="Total day minutes", ge=0)
    Total_day_calls: int = Field(..., alias="Total day calls", ge=0)
    Total_day_charge: float = Field(..., alias="Total day charge", ge=0)
    Total_eve_minutes: float = Field(..., alias="Total eve minutes", ge=0)
    Total_eve_calls: int = Field(..., alias="Total eve calls", ge=0)
    Total_eve_charge: float = Field(..., alias="Total eve charge", ge=0)
    Total_night_minutes: float = Field(..., alias="Total night minutes", ge=0)
    Total_night_calls: int = Field(..., alias="Total night calls", ge=0)
    Total_night_charge: float = Field(..., alias="Total night charge", ge=0)
    Total_intl_minutes: float = Field(..., alias="Total intl minutes", ge=0)
    Total_intl_calls: int = Field(..., alias="Total intl calls", ge=0)
    Total_intl_charge: float = Field(..., alias="Total intl charge", ge=0)
    Customer_service_calls: int = Field(..., alias="Customer service calls", ge=0)


def _prepare_features(data: CustomerInput) -> pd.DataFrame:
    if MODEL is None or SCALER is None or EXPECTED_FEATURES is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded yet. Train the model first.")

    input_dict = data.dict(by_alias=True)
    df = pd.DataFrame([input_dict])
    df["International plan_Yes"] = df["International plan"].map({"yes": 1, "no": 0})
    df["Voice mail plan_Yes"] = df["Voice mail plan"].map({"yes": 1, "no": 0})
    df = pd.get_dummies(df, columns=["State"], prefix="State")
    df = df.reindex(columns=EXPECTED_FEATURES, fill_value=0)

    numeric_columns = [
        column
        for column in EXPECTED_FEATURES
        if column not in {"International plan_Yes", "Voice mail plan_Yes"} and not column.startswith("State_")
    ]
    df[numeric_columns] = SCALER.transform(df[numeric_columns])
    return df


@app.post(
    "/predict/",
    summary="Predict Customer Churn",
    description="Predicts whether a customer is likely to churn based on service usage patterns",
    response_description="Churn prediction result",
    tags=["Predictions"],
)
async def predict(data: CustomerInput):
    try:
        df = _prepare_features(data)
        prediction = MODEL.predict(df)
        result = bool(prediction[0])

        with mlflow.start_run():
            run_id = mlflow.active_run().info.run_id
            mlflow.log_param("account_length", data.Account_length)
            mlflow.log_metric("churn_prediction", int(result))
            log_mlflow_to_es(run_id, "churn_prediction", int(result))

        return {"churn_prediction": result}
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Customer Churn Prediction API - visit /docs for interactive documentation"}
