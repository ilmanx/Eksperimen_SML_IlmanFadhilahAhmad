import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load dataset
def load_data(input_path="dataset_raw/Data_Breast_Cancer_raw.csv"):
    """
    Load dataset dari path yang diberikan.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")
    df = pd.read_csv(input_path)
    return df

# preprocessing
def preprocess_data(df):
    """
    Drop kolom yang tidak diperlukan dan encode target variable.
    """
    df = df.drop(columns=["id", "Unnamed: 32"], errors='ignore')
    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})
    return df

# split dan scaling
def split_and_scale(df, test_size=0.2, random_state=42):
    """
    Split dataset menjadi train/test dan lakukan scaling.
    """
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

# menyimpan dataset
def save_data(X_train_scaled, X_test_scaled, y_train, y_test, columns, output_dir="preprocessing/dataset_preprocessing"):
    """
    Simpan dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    train_data = pd.concat(
        [pd.DataFrame(X_train_scaled, columns=columns), y_train.reset_index(drop=True)],
        axis=1
    )

    test_data = pd.concat(
        [pd.DataFrame(X_test_scaled, columns=columns), y_test.reset_index(drop=True)],
        axis=1
    )

    train_data.to_csv(f"{output_dir}/train_data.csv", index=False)
    test_data.to_csv(f"{output_dir}/test_data.csv", index=False)
    print(f" Dataset preprocessing berhasil disimpan di folder: {output_dir}")

# run preprocessing
def run_preprocessing(input_path="dataset_raw/Data_Breast_Cancer_raw.csv",
                      output_dir="preprocessing/dataset_preprocessing"):

    df = load_data(input_path)
    df = preprocess_data(df)
    X_train_scaled, X_test_scaled, y_train, y_test, columns = split_and_scale(df)
    save_data(X_train_scaled, X_test_scaled, y_train, y_test, columns, output_dir)

    print("Preprocessing selesai!")

if __name__ == "__main__":
    run_preprocessing()
