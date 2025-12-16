import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
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

# load data
train_df = pd.read_csv("../preprocessing/train_data.csv")
test_df = pd.read_csv("../preprocessing/test_data.csv")

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# hypter tuning
param_grid = [
    {"C": 0.1},
    {"C": 1.0},
    {"C": 10.0}
]

for params in param_grid:
    with mlflow.start_run(run_name=f"LogReg_C_{params['C']}"):

        # Model
        model = LogisticRegression(
            C=params["C"],
            solver="liblinear",
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # 🔹 Manual logging
        mlflow.log_param("C", params["C"])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        # Classification Report
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # Metrics Summary (extra, optional tapi aman)
        metrics_summary = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "C": params["C"]
        }
        with open("metrics_summary.json", "w") as f:
            json.dump(metrics_summary, f, indent=4)
        mlflow.log_artifact("metrics_summary.json")

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"C={params['C']} | Accuracy={acc:.4f}")