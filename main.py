import argparse
import os
import pandas as pd
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Ensure a local directory for MLflow artifacts exists
os.makedirs("mlruns", exist_ok=True)
artifact_location = os.path.abspath("mlruns")

# Set MLflow tracking URI to use a file-based store in the mlruns directory
mlflow.set_tracking_uri("file://" + artifact_location)

# Configure experiment (this will create a new experiment if one doesn't exist)
experiment_name = "Churn_Prediction"
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
else:
    experiment_id = experiment.experiment_id
mlflow.set_experiment(experiment_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate", "load", "prepare"], required=True,
                        help="Mode: train, evaluate, load, or prepare")
    parser.add_argument("--train_data", help="Path to training data")
    parser.add_argument("--test_data", help="Path to test data")
    parser.add_argument("--output", help="Path to save prepared data")
    parser.add_argument("--save", help="Path to save model", default="model.pkl")
    parser.add_argument("--load", help="Path to load model")
    args = parser.parse_args()

    print("Arguments parsed:", args)

    if args.mode == "prepare":
        if not args.train_data or not args.test_data or not args.output:
            print("Error: --train_data, --test_data, and --output are required in prepare mode.")
            return

        print("Preparing data...")
        df_train, df_test, scaler = prepare_data(args.train_data, args.test_data)
        
        # Save prepared data for later use
        df_train.to_pickle(args.output)
        df_test.to_pickle(args.output.replace(".pkl", "_test.pkl"))
        print(f"Prepared data saved to {args.output}")

    elif args.mode == "train":
        if not args.train_data or not args.test_data:
            print("Error: --train_data and --test_data are required in train mode.")
            return

        print("Preparing data...")
        df_train, df_test, scaler = prepare_data(args.train_data, args.test_data)
        
        X_train, y_train = df_train.drop(columns=['Churn']), df_train['Churn']
        X_test, y_test = df_test.drop(columns=['Churn']), df_test['Churn']
        
        # Start a new MLflow run
        with mlflow.start_run():
            mlflow.log_param("model_type", "DecisionTreeClassifier")
            
            print("Training decision tree model...")
            model = train_model(X_train, y_train)
            
            print("Evaluating model...")
            acc, auc, report, matrix = evaluate_model(model, X_test, y_test)
            
            print("Accuracy:", acc)
            print("AUC Score:", auc)
            print("Classification Report:\n", report)
            print("Confusion Matrix:\n", matrix)
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("auc", auc)
            
            # Provide a one-row DataFrame as the input example
            input_example = X_train.iloc[0:1]
            mlflow.sklearn.log_model(model, "model", input_example=input_example)
            
            # Save and log the classification report as an artifact
            with open("classification_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("classification_report.txt")
            
            print(f"Saving model to {args.save}...")
            save_model(model, scaler, args.save)
            print(f"Model saved to {args.save}")

    elif args.mode == "evaluate":
        if not args.load or not args.test_data:
            print("Error: --load and --test_data are required in evaluate mode.")
            return

        print(f"Loading model from {args.load}...")
        model, scaler = load_model(args.load)
        print("Model loaded successfully!")
        
        print("Preparing test data...")
        # Assumes prepare_data can handle None for train_data when a scaler is provided
        _, df_test, _ = prepare_data(None, args.test_data, scaler)
        
        X_test = df_test.drop(columns=['Churn'])
        y_test = df_test['Churn']
        
        acc, auc, report, matrix = evaluate_model(model, X_test, y_test)
        print("Accuracy:", acc)
        print("AUC Score:", auc)
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", matrix)
        
        with mlflow.start_run():
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("auc", auc)
            with open("classification_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("classification_report.txt")

    elif args.mode == "load":
        if not args.load:
            print("Error: --load is required in load mode.")
            return

        print(f"Loading model from {args.load}...")
        model, scaler = load_model(args.load)
        print("Model loaded successfully!")

    else:
        print("Invalid mode. Use --mode train, evaluate, load, or prepare.")

if __name__ == "__main__":
    main()
