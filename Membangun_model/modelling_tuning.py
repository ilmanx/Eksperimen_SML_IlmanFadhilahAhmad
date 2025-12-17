import mlflow
import dagshub
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import warnings
warnings.filterwarnings("ignore")

# DAGSHUB
dagshub.init(
    repo_owner="ilmanx",
    repo_name="Eksperimen_SML_IlmanFadhilahAhmad",
    mlflow=True
)

mlflow.set_experiment("Breast_Cancer_Classification")

# 2. LOAD DATA
data = pd.read_csv("../preprocessing/train_data.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

input_example = X_train.iloc[:5]

# HYPER TUNING
C_range = np.linspace(0.01, 10, 5)

for C in C_range:
    with mlflow.start_run(run_name=f"logreg_C_{C:.2f}"):

        # MODEL
        model = LogisticRegression(
            C=C,
            solver="liblinear",
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # MANUAL METRICS
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # LOG PARAMETERS
        mlflow.log_param("C", C)
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("random_state", 42)

        # LOG METRICS
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # list artifact
        #Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 4))
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        #Classification Report
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        #Metrics Summary
        metrics_summary = {
            "C": C,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }
        with open("metrics_summary.json", "w") as f:
            json.dump(metrics_summary, f, indent=4)
        mlflow.log_artifact("metrics_summary.json")

        # LOG MODEL
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        print(f"C={C:.2f} | Accuracy={acc:.4f}")

print("Training selesai (DagsHub)")
