import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Breast_Cancer_Classification_HyperTuning")

# Load Data
data = pd.read_csv("../preprocessing/train_data.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.2
)

input_example = X_train.iloc[:5]

# hyperparameter
C_range = np.linspace(0.01, 10, 5)

best_accuracy = 0
best_params = {}

# loop
for C in C_range:
    with mlflow.start_run(run_name=f"logreg_C_{C}"):

        mlflow.autolog()

        model = LogisticRegression(
            C=C,
            solver="liblinear",
            random_state=42
        )
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {"C": C}

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example
            )

        print(f"C={C} | Accuracy={accuracy:.4f}")

print("\nBest Accuracy :", best_accuracy)
print("Best Params   :", best_params)