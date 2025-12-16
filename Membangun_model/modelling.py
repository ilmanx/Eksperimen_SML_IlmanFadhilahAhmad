"""
Modelling - Kriteria 2 (Basic)
Training Machine Learning menggunakan MLflow Autolog
Dataset: Breast Cancer (Preprocessed)
"""

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    # ===============================
    # 1. Load preprocessed dataset
    # ===============================
    train_path = "../preprocessing/train_data.csv"
    test_path = "../preprocessing/test_data.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # ===============================
    # 2. Split feature & target
    # ===============================
    X_train = train_df.drop(columns=["diagnosis"])
    y_train = train_df["diagnosis"]

    X_test = test_df.drop(columns=["diagnosis"])
    y_test = test_df["diagnosis"]

    # ===============================
    # 3. Setup MLflow (local)
    # ===============================
    mlflow.set_experiment("Breast_Cancer_Classification")
    mlflow.sklearn.autolog()

    # ===============================
    # 4. Train model
    # ===============================
    with mlflow.start_run(run_name="LogisticRegression"):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # ===============================
        # 5. Evaluation
        # ===============================
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("Model training selesai")
        print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
