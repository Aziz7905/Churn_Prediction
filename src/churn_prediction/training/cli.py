from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn

from churn_prediction.config import (
    DEFAULT_CLASSIFICATION_REPORT_PATH,
    DEFAULT_EVALUATION_RESULTS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREPARED_DATA_PATH,
    RAW_DATA_DIR,
    TEST_DATA_FILENAME,
    TRAIN_DATA_FILENAME,
    ensure_directories,
    mlflow_tracking_uri,
)
from churn_prediction.training.core import evaluate_model, load_model, prepare_data, resolve_target_column, save_model, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the churn prediction model.")
    parser.add_argument("--mode", choices=["train", "evaluate", "load", "prepare"], required=True)
    parser.add_argument("--train-data", default=str(RAW_DATA_DIR / TRAIN_DATA_FILENAME))
    parser.add_argument("--test-data", default=str(RAW_DATA_DIR / TEST_DATA_FILENAME))
    parser.add_argument("--output", default=str(DEFAULT_PREPARED_DATA_PATH))
    parser.add_argument("--save", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--load", default=str(DEFAULT_MODEL_PATH))
    return parser


def _log_report(report: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    mlflow.log_artifact(str(output_path))


def _save_prepared_datasets(df_train, df_test, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_output_path = output_path.with_name(f"{output_path.stem}_test{output_path.suffix}")
    df_train.to_pickle(output_path)
    df_test.to_pickle(test_output_path)


def main(argv: list[str] | None = None) -> None:
    ensure_directories()
    mlflow.set_tracking_uri(mlflow_tracking_uri())
    mlflow.set_experiment("Churn_Prediction")

    parser = build_parser()
    args = parser.parse_args(argv)

    train_data = Path(args.train_data)
    test_data = Path(args.test_data)
    output_path = Path(args.output)
    model_path = Path(args.save)
    load_path = Path(args.load)

    if args.mode == "prepare":
        df_train, df_test, _ = prepare_data(train_data, test_data)
        _save_prepared_datasets(df_train, df_test, output_path)
        print(f"Prepared data saved to {output_path}")
        return

    if args.mode == "train":
        df_train, df_test, scaler = prepare_data(train_data, test_data)
        target_column = resolve_target_column(df_train.columns.tolist())
        X_train = df_train.drop(columns=[target_column])
        y_train = df_train[target_column]
        X_test = df_test.drop(columns=[target_column])
        y_test = df_test[target_column]

        with mlflow.start_run():
            mlflow.log_param("model_type", "DecisionTreeClassifier")
            model = train_model(X_train, y_train)
            acc, auc, report, matrix = evaluate_model(model, X_test, y_test)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("auc", auc)
            mlflow.sklearn.log_model(model, "model", input_example=X_train.iloc[0:1])

            report_path = DEFAULT_CLASSIFICATION_REPORT_PATH
            report_path.parent.mkdir(parents=True, exist_ok=True)
            _log_report(report, report_path)
            DEFAULT_EVALUATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            DEFAULT_EVALUATION_RESULTS_PATH.write_text(
                f"Accuracy: {acc}\nAUC: {auc}\nConfusion matrix:\n{matrix}\n",
                encoding="utf-8",
            )

            save_model(model, scaler, model_path)
            print(f"Model saved to {model_path}")
            print(f"Accuracy: {acc}")
            print(f"AUC Score: {auc}")
            print("Classification Report:\n", report)
            print("Confusion Matrix:\n", matrix)
        return

    if args.mode == "evaluate":
        model, scaler = load_model(load_path)
        _, df_test, _ = prepare_data(None, test_data, scaler)
        target_column = resolve_target_column(df_test.columns.tolist())
        X_test = df_test.drop(columns=[target_column])
        y_test = df_test[target_column]
        acc, auc, report, matrix = evaluate_model(model, X_test, y_test)

        with mlflow.start_run():
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("auc", auc)
            _log_report(report, DEFAULT_CLASSIFICATION_REPORT_PATH)
            DEFAULT_EVALUATION_RESULTS_PATH.write_text(
                f"Accuracy: {acc}\nAUC: {auc}\nConfusion matrix:\n{matrix}\n",
                encoding="utf-8",
            )

        print(f"Accuracy: {acc}")
        print(f"AUC Score: {auc}")
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", matrix)
        return

    if args.mode == "load":
        model, scaler = load_model(load_path)
        print(f"Model loaded from {load_path}")
        print(f"Model type: {type(model).__name__}")
        print(f"Scaler type: {type(scaler).__name__}")
        return


if __name__ == "__main__":
    main()
