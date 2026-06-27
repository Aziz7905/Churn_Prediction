# Customer Churn MLOps Project

An end-to-end machine learning project for customer churn prediction with a small MLOps workflow:

- data preparation and model training
- evaluation with saved reports
- FastAPI inference service
- browser-based frontend
- MLflow tracking
- optional Elasticsearch logging for prediction metrics

## Project Structure

```text
.
├── api.py
├── frontend.py
├── train.py
├── data/
│   └── raw/
├── artifacts/
│   ├── models/
│   ├── processed/
│   └── reports/
├── src/
│   └── churn_prediction/
│       ├── api/
│       ├── frontend/
│       └── pipeline/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── .github/workflows/
```

## What This Project Does

- Trains a `DecisionTreeClassifier` on the churn dataset.
- Preprocesses categorical and numeric features consistently.
- Stores model and preprocessing artifacts together.
- Exposes a prediction API with FastAPI.
- Serves a simple UI for manual testing.
- Logs training metrics and prediction metrics with MLflow.

## Data

The raw datasets live in `data/raw/`:

- `churn-bigml-80.csv`
- `churn-bigml-20.csv`

These are used for training and evaluation.

## Main Entry Points

- `train.py` - training, evaluation, data preparation
- `api.py` - FastAPI prediction service
- `frontend.py` - browser UI

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

```bash
python train.py --mode prepare
```

### 3. Train the model

```bash
python train.py --mode train
```

### 4. Evaluate the model

```bash
python train.py --mode evaluate
```

### 5. Run the API

```bash
uvicorn api:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

### 6. Run the frontend

```bash
uvicorn frontend:app --reload --port 8001
```

Open the UI in your browser and submit sample customer data.
By default, the frontend runs at `http://localhost:8001`.

## MLflow

By default, MLflow artifacts are stored locally under `artifacts/mlruns/`.

You can override the tracking backend by setting `MLFLOW_TRACKING_URI` before running training or the API.

## Optional Elasticsearch Logging

Prediction metrics can also be sent to Elasticsearch if it is available.

Set `ELASTICSEARCH_URL` if you want to point the API at a different Elasticsearch instance.

If Elasticsearch is unavailable, the API continues to work and only skips that extra logging step.

## Makefile Commands

- `make prepare`
- `make train`
- `make evaluate`
- `make build`
- `make run`

If you want the Docker image to serve predictions immediately, run `make train` first so `artifacts/models/churn_model.pkl` exists before building the image.

## Notes for GitHub

- Generated files are ignored via `.gitignore`.
- The reusable code now lives under `src/churn_prediction/`.
- The old prototype filenames were replaced by clearer entry points.

## Future Improvements

- add a stricter train/test split workflow
- add automated unit tests for preprocessing
- add a lightweight model registry pattern
- split the UI and API into separate deployable services
