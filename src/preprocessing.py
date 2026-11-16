import os
import pandas as pd

from sklearn.model_selection import train_test_split


def prepare_data(input_path, output_dir):

    df = pd.read_csv(input_path)

    target = "Mental_Health_Risk"
    id_column = "Student_ID"

    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found")

    df = df.dropna(subset=[target])

    X = df.drop(columns=[target, id_column], errors="ignore")
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print("Data preprocessing completed")

    return output_dir