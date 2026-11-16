import os
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, RandomizedSearchCV , StratifiedKFold


def train_models(processed_path, model_dir, report_dir):

    X_train = pd.read_csv(f"{processed_path}/X_train.csv")
    y_train = pd.read_csv(f"{processed_path}/y_train.csv").squeeze() # transforme DataFrame a une seule colonne en Series

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    numeric_features = X_train.select_dtypes(include="number").columns
    categorical_features = X_train.select_dtypes(exclude="number").columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000)
    }

    results = {}
    best_score = 0
    best_model = None

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessing", preprocessing),
            ("model", model)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring="f1_macro"
        )

        results[name] = {
            "mean_f1": float(scores.mean()),
            "std_f1": float(scores.std())
        }

        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = pipeline

    lr_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", LogisticRegression(max_iter=2000))
    ])

    parameters = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__class_weight": [None, "balanced"],
        "model__solver": ["lbfgs"]
    }

    search = RandomizedSearchCV(
        lr_pipeline,
        parameters,
        n_iter=10,
        cv=skf,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    results["Tuned Logistic Regression"] = {
        "mean_f1": float(search.best_score_)
    }

    if search.best_score_ > best_score:
        best_model = search.best_estimator_

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    model_path = f"{model_dir}/best_pipeline.joblib"

    joblib.dump(best_model, model_path)

    with open(f"{report_dir}/training_results.json", "w") as file:
        json.dump(results, file, indent=4)

    print("Training completed")
    print("Best model:", type(best_model.named_steps["model"]).__name__)

    return model_path