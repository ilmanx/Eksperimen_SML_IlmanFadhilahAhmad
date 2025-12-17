import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# MLflow setup
mlflow.set_experiment("Breast_Cancer_Classification")

# autolog
mlflow.sklearn.autolog()

# Load data
train_df = pd.read_csv("../preprocessing/train_data.csv")
test_df = pd.read_csv("../preprocessing/test_data.csv")

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

with mlflow.start_run(run_name="LogisticRegression_Basic"):

    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)
    with open("estimator.html", "w") as f:
        f.write(f"<pre>{report}</pre>")

    mlflow.log_artifact("estimator.html")

    print(f"Training selesai | Accuracy: {acc:.4f}")