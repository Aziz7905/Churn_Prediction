# Customer Churn MLOps Project

An end-to-end machine learning project for customer churn prediction with a small MLOps workflow:

- data preparation and model training
- evaluation with saved reports
- FastAPI inference service
- browser-based frontend
- MLflow tracking
- optional Elasticsearch logging for prediction metrics
- optional Kibana dashboard for inspecting Elasticsearch data

## Project Structure

```text
.
├── run_training.py
├── serve_api.py
├── serve_web.py
├── data/
│   └── raw/
├── artifacts/
│   ├── models/
│   ├── processed/
│   └── reports/
├── src/
│   └── churn_prediction/
│       ├── serving/
│       ├── training/
│       └── web/
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

- `run_training.py` - training, evaluation, data preparation
- `serve_api.py` - FastAPI prediction service
- `serve_web.py` - browser UI

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Windows Command Prompt: `.\.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare data

```bash
python run_training.py --mode prepare
```

### 4. Train the model

```bash
python run_training.py --mode train
```

### 5. Evaluate the model

```bash
python run_training.py --mode evaluate
```

### 6. Run the API

```bash
uvicorn serve_api:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

### 7. Run the frontend

```bash
uvicorn serve_web:app --reload --port 8001
```

Open the UI in your browser and submit sample customer data.
By default, the frontend runs at `http://localhost:8001`.

## MLflow

By default, MLflow uses a local SQLite tracking database at `artifacts/mlflow.db`.

You can override the tracking backend by setting `MLFLOW_TRACKING_URI` before running training or the API.

## Optional Elasticsearch Logging

Prediction metrics can also be sent to Elasticsearch if it is available.

Set `ELASTICSEARCH_URL` if you want to point the API at a different Elasticsearch instance.

If Elasticsearch is unavailable, the API continues to work and only skips that extra logging step.

## Elasticsearch and Kibana

The `docker-compose.yml` file starts both Elasticsearch and Kibana together.

- Elasticsearch runs on `http://localhost:9200`
- Kibana runs on `http://localhost:5601`

Use Kibana to explore the metrics index that the API writes to when Elasticsearch logging is enabled.

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
