.PHONY: prepare train evaluate clean

# Variables
TRAIN_DATA = churn-bigml-80.csv
TEST_DATA = churn-bigml-20.csv
OUTPUT = prepared_data.pkl
MODEL = model.pkl
IMAGE_NAME=aziz_barkaoui_ds1_mlop
DOCKER_USER=aziz7905
PORT=8000

# Prepare data: this saves the processed training and test data to pickle files.
prepare:
	python3 main.py --mode prepare --train_data $(TRAIN_DATA) --test_data $(TEST_DATA) --output $(OUTPUT)

# Train model: reads the raw CSVs, processes them, trains the model, and saves it.
train:
	python3 main.py --mode train --train_data $(TRAIN_DATA) --test_data $(TEST_DATA) --save $(MODEL)

# Evaluate model: loads the saved model and scaler, processes the test data using the saved scaler, and evaluates.
evaluate:
	python3 main.py --mode evaluate --load $(MODEL) --test_data $(TEST_DATA)

# Clean: remove generated pickle and model files.
clean:
	rm -f $(OUTPUT) $(MODEL) *.pkl

# Run all steps
all: prepare train evaluate

# Construction de l'image
build:
	docker build -t $(IMAGE_NAME) .

# Exécution du conteneur
run:
	docker run -d -p $(PORT):8000 $(IMAGE_NAME)
