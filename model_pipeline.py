import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score
import mlflow
import mlflow.sklearn


def prepare_data(train_path, test_path, scaler=None):
    """
    If train_path is provided, prepare both training and test data and fit/transform the scaler.
    If train_path is None, only process the test data using the provided scaler.
    """
    if train_path is not None:
        # Read both training and test data
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        print("Initial training data types:")
        print(df_train.dtypes)  # Check initial column types
        
        # One-hot encoding for both datasets
        df_train = pd.get_dummies(df_train, drop_first=True)
        df_test = pd.get_dummies(df_test, drop_first=True)
        
        # Align test data columns with training data
        df_test = df_test.reindex(columns=df_train.columns, fill_value=0)
        
        # Get numeric columns
        num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Scale the data
        if scaler is None:
            scaler = StandardScaler()
            df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
        else:
            df_train[num_cols] = scaler.transform(df_train[num_cols])
        df_test[num_cols] = scaler.transform(df_test[num_cols])
        
        print("Training data after scaling:")
        print(df_train.head())
        
        return df_train, df_test, scaler
    else:
        # Only test data is provided
        df_test = pd.read_csv(test_path)
        df_test = pd.get_dummies(df_test, drop_first=True)
        if scaler is not None:
            # We assume that the test data already has the same columns as used during training.
            num_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
            df_test[num_cols] = scaler.transform(df_test[num_cols])
        return None, df_test, scaler

def train_model(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    return acc, auc, report, matrix

def save_model(model, scaler, filename="model.pkl"):
    joblib.dump((model, scaler), filename)

def load_model(filename="model.pkl"):
    print(f"Loading model from {filename}...")
    return joblib.load(filename)
