import os
import json
import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


def evaluate_model(processed_path, model_path, report_dir):

    X_test = pd.read_csv(f"{processed_path}/X_test.csv")
    y_test = pd.read_csv(f"{processed_path}/y_test.csv").squeeze()

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    results = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(
            y_test, y_pred, average="macro", zero_division=0
        )),
        "recall": float(recall_score(
            y_test, y_pred, average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        )
    }

    os.makedirs(report_dir, exist_ok=True)

    with open(f"{report_dir}/metrics.json", "w") as file:
        json.dump(results, file, indent=4)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues"
    )

    plt.title("StudentMind - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{report_dir}/confusion_matrix.png")
    plt.close()

    print("Evaluation completed")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"F1 Macro: {results['f1_macro']:.2%}")

    return results