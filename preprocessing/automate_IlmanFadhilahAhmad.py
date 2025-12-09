import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_preprocessing(
        input_path="dataset_raw/Data_Breast_Cancer_raw.csv",
        output_dir="preprocessing"
    ):

    print(" Preprocessing Otomatis ")

    df = pd.read_csv(input_path)

    df = df.drop(columns=["id", "Unnamed: 32"])
    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(output_dir, exist_ok=True)

    train_data = pd.concat(
        [pd.DataFrame(X_train_scaled, columns=X.columns), y_train.reset_index(drop=True)],
        axis=1
    )

    test_data = pd.concat(
        [pd.DataFrame(X_test_scaled, columns=X.columns), y_test.reset_index(drop=True)],
        axis=1
    )

    train_data.to_csv(f"{output_dir}/train_data.csv", index=False)
    test_data.to_csv(f"{output_dir}/test_data.csv", index=False)

    print(" Preprocessing Berhasil")

if __name__ == "__main__":
    run_preprocessing()
