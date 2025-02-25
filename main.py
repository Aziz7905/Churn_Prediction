import argparse
import pandas as pd
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model
import mlflow
import mlflow.sklearn

# Configuration de MLflow : définir le backend (ici SQLite) et l'expérience
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Churn_Prediction")

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
        
        # Sauvegarde des données préparées pour une utilisation ultérieure
        df_train.to_pickle(args.output)  # Sauvegarde des données d'entraînement
        df_test.to_pickle(args.output.replace(".pkl", "_test.pkl"))  # Sauvegarde des données de test
        print(f"Prepared data saved to {args.output}")

    elif args.mode == "train":
        if not args.train_data or not args.test_data:
            print("Error: --train_data and --test_data are required in train mode.")
            return

        print("Preparing data...")
        df_train, df_test, scaler = prepare_data(args.train_data, args.test_data)
        
        X_train, y_train = df_train.drop(columns=['Churn']), df_train['Churn']
        X_test, y_test = df_test.drop(columns=['Churn']), df_test['Churn']
        
        # Démarrage d'une nouvelle exécution MLflow
        with mlflow.start_run():
            # Enregistrement des hyperparamètres
            mlflow.log_param("model_type", "DecisionTreeClassifier")
            # Vous pouvez ajouter d'autres hyperparamètres ici, par exemple :
            # mlflow.log_param("max_depth", None)
            
            print("Training decision tree model...")
            model = train_model(X_train, y_train)
            
            print("Evaluating model...")
            acc, auc, report, matrix = evaluate_model(model, X_test, y_test)
            
            print("Accuracy:", acc)
            print("AUC Score:", auc)
            print("Classification Report:\n", report)
            print("Confusion Matrix:\n", matrix)
            
            # Enregistrement des métriques
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("auc", auc)
            
            # Enregistrement du modèle via MLflow
            mlflow.sklearn.log_model(model, "model")
            
            # Enregistrement d'un artifact contenant le rapport de classification
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
        # On passe None pour train_data et on fournit le scaler chargé
        _, df_test, _ = prepare_data(None, args.test_data, scaler)
        
        X_test = df_test.drop(columns=['Churn'])
        y_test = df_test['Churn']
        
        acc, auc, report, matrix = evaluate_model(model, X_test, y_test)
        print("Accuracy:", acc)
        print("AUC Score:", auc)
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", matrix)
        
        # Enregistrement des métriques d'évaluation dans MLflow
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

