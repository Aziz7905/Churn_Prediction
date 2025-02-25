from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import mlflow
from elasticsearch import Elasticsearch
import logging

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using a pre-trained Decision Tree model",
    version="1.0.0",
    contact={
        "name": "Aziz Allah Barkaoui",
        "email": "aziz@example.com",
    },
    license_info={
        "name": "MIT",
    },
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set MLflow tracking URI (pointing to our MLflow server)
mlflow.set_tracking_uri("http://localhost:5000")

# Connect to Elasticsearch (assurez-vous que ES est lancé)
es = Elasticsearch(["http://localhost:9200"])

# Configure logging
logger = logging.getLogger("churn_api")
logger.setLevel(logging.INFO)

# Load model, scaler, and training columns
model, scaler = joblib.load("model.pkl")
prepared_data = pd.read_pickle("prepared_data.pkl")
expected_features = prepared_data.drop(columns=['Churn']).columns.tolist()

def log_mlflow_to_es(run_id, metric_name, value):
    """
    Envoie une métrique de MLflow vers Elasticsearch.
    Les documents seront indexés dans l'index "mlflow-metrics".
    """
    doc = {
        "run_id": run_id,
        "metric_name": metric_name,
        "value": value
    }
    es.index(index="mlflow-metrics", body=doc)

class CustomerInput(BaseModel):
    State: str = Field(..., 
                         alias="State",
                         description="US State abbreviation",
                         example="LA")
    
    Account_length: int = Field(...,
                                alias="Account length",
                                ge=0,
                                description="Number of months the customer has been with the company",
                                example=117)
    
    Area_code: int = Field(...,
                           alias="Area code",
                           description="Area code number",
                           example=408)
    
    International_plan: str = Field(...,
                                    alias="International plan",
                                    description="Whether the customer has an international plan",
                                    example="no",
                                    pattern="^(yes|no)$")
    
    Voice_mail_plan: str = Field(...,
                                 alias="Voice mail plan",
                                 description="Whether the customer has a voice mail plan",
                                 example="no",
                                 pattern="^(yes|no)$")
    
    Number_vmail_messages: int = Field(...,
                                       alias="Number vmail messages",
                                       ge=0,
                                       description="Number of voice mail messages",
                                       example=0)
    
    Total_day_minutes: float = Field(...,
                                     alias="Total day minutes",
                                     ge=0,
                                     description="Total minutes of day calls",
                                     example=184.5)
    
    Total_day_calls: int = Field(...,
                                 alias="Total day calls",
                                 ge=0,
                                 description="Total number of day calls",
                                 example=97)
    
    Total_day_charge: float = Field(...,
                                    alias="Total day charge",
                                    ge=0,
                                    description="Total charge for day calls",
                                    example=31.37)
    
    Total_eve_minutes: float = Field(...,
                                     alias="Total eve minutes",
                                     ge=0,
                                     description="Total minutes of evening calls",
                                     example=351.6)
    
    Total_eve_calls: int = Field(...,
                                 alias="Total eve calls",
                                 ge=0,
                                 description="Total number of evening calls",
                                 example=80)
    
    Total_eve_charge: float = Field(...,
                                    alias="Total eve charge",
                                    ge=0,
                                    description="Total charge for evening calls",
                                    example=29.89)
    
    Total_night_minutes: float = Field(...,
                                       alias="Total night minutes",
                                       ge=0,
                                       description="Total minutes of night calls",
                                       example=215.8)
    
    Total_night_calls: int = Field(...,
                                   alias="Total night calls",
                                   ge=0,
                                   description="Total number of night calls",
                                   example=90)
    
    Total_night_charge: float = Field(...,
                                      alias="Total night charge",
                                      ge=0,
                                      description="Total charge for night calls",
                                      example=9.71)
    
    Total_intl_minutes: float = Field(...,
                                      alias="Total intl minutes",
                                      ge=0,
                                      description="Total minutes of international calls",
                                      example=8.7)
    
    Total_intl_calls: int = Field(...,
                                  alias="Total intl calls",
                                  ge=0,
                                  description="Total number of international calls",
                                  example=4)
    
    Total_intl_charge: float = Field(...,
                                     alias="Total intl charge",
                                     ge=0,
                                     description="Total charge for international calls",
                                     example=2.35)
    
    Customer_service_calls: int = Field(...,
                                        alias="Customer service calls",
                                        ge=0,
                                        description="Number of calls to customer service",
                                        example=1)

@app.post("/predict/",
          summary="Predict Customer Churn",
          description="Predicts whether a customer is likely to churn based on their service usage patterns",
          response_description="Churn Prediction Result",
          tags=["Predictions"])
async def predict(data: CustomerInput):
    """
    Effectue une prédiction de churn pour un client en se basant sur ses données d'utilisation.
    Enregistre également la métrique dans MLflow et l'envoie vers Elasticsearch.
    """
    try:
        input_dict = data.dict(by_alias=True)
        df = pd.DataFrame([input_dict])
        
        # Encodage manuel des variables catégorielles
        df['International plan_Yes'] = df['International plan'].map({'yes': 1, 'no': 0})
        df['Voice mail plan_Yes'] = df['Voice mail plan'].map({'yes': 1, 'no': 0})
        
        # Encodage one-hot pour la variable State
        df = pd.get_dummies(df, columns=['State'], prefix='State')
        
        # Re-indexer pour correspondre aux colonnes utilisées lors de l'entraînement
        df = df.reindex(columns=expected_features, fill_value=0)
        
        # Mise à l'échelle des variables numériques
        num_cols = [col for col in expected_features 
                    if col not in ['International plan_Yes', 'Voice mail plan_Yes'] and not col.startswith('State_')]
        df[num_cols] = scaler.transform(df[num_cols])
        
        # Effectuer la prédiction
        prediction = model.predict(df)
        result = bool(prediction[0])
        
        # Enregistrer la prédiction avec MLflow et envoyer les logs vers Elasticsearch
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_param("Account length", data.Account_length)
            mlflow.log_metric("churn_prediction", int(result))
            # Envoi du log vers Elasticsearch (index mlflow-metrics)
            log_mlflow_to_es(run_id, "churn_prediction", int(result))
        
        return {"churn_prediction": result}
    
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Customer Churn Prediction API - Visit /docs for interactive documentation"}

