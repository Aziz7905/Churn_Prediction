.PHONY: prepare train evaluate clean build run run-api run-frontend

PYTHON ?= python
TRAIN_DATA = data/raw/churn-bigml-80.csv
TEST_DATA = data/raw/churn-bigml-20.csv
OUTPUT = artifacts/processed/prepared_data.pkl
MODEL = artifacts/models/churn_model.pkl
IMAGE_NAME = customer-churn-mlops
PORT = 8000

prepare:
	$(PYTHON) train.py --mode prepare --train-data $(TRAIN_DATA) --test-data $(TEST_DATA) --output $(OUTPUT)

train:
	$(PYTHON) train.py --mode train --train-data $(TRAIN_DATA) --test-data $(TEST_DATA) --save $(MODEL)

evaluate:
	$(PYTHON) train.py --mode evaluate --test-data $(TEST_DATA) --load $(MODEL)

run-api:
	uvicorn api:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	uvicorn frontend:app --reload --host 0.0.0.0 --port 8001

clean:
	rm -rf artifacts

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -d -p $(PORT):8000 -p 8001:8001 -p 5002:5002 $(IMAGE_NAME)
