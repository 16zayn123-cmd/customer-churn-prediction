from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.download_data import ensure_data
except ModuleNotFoundError:
    from download_data import ensure_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_clean_data() -> pd.DataFrame:
    data_path = ensure_data()
    df = pd.read_csv(data_path)

    # TotalCharges contains blank strings for a small number of new customers.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Remove exact duplicate rows if any are present.
    df = df.drop_duplicates().copy()

    # Normalise the target.
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})

    if df[TARGET].isna().any():
        raise ValueError("Unexpected values found in the Churn target column.")

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop=None,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    pipelines = {}
    for name, classifier in models.items():
        pipelines[name] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", classifier),
            ]
        )
    return pipelines


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def save_confusion_matrix(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Stayed", "Churned"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Selected Model")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def get_global_feature_importance(model: Pipeline, top_n: int = 15) -> list[dict]:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "coef_"):
        raw_values = estimator.coef_[0]
        importance = np.abs(raw_values)
        signed_values = raw_values
    elif hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
        signed_values = estimator.feature_importances_
    else:
        return []

    indices = np.argsort(importance)[::-1][:top_n]
    result = []
    for idx in indices:
        result.append(
            {
                "feature": str(feature_names[idx]),
                "importance": float(importance[idx]),
                "signed_value": float(signed_values[idx]),
            }
        )
    return result


def train_project() -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_data()

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    all_metrics = {}
    trained_models = {}

    for name, pipeline in build_models().items():
        print(f"Training: {name}")
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        all_metrics[name] = metrics
        trained_models[name] = pipeline
        print(json.dumps(metrics, indent=2))

    # Select by ROC-AUC because the output is used as a risk ranking/probability.
    best_name = max(all_metrics, key=lambda name: all_metrics[name]["roc_auc"])
    best_model = trained_models[best_name]

    joblib.dump(best_model, MODEL_PATH)
    save_confusion_matrix(best_model, X_test, y_test)

    metadata = {
        "selected_model": best_name,
        "selection_metric": "roc_auc",
        "decision_threshold": 0.5,
        "dataset_rows_after_cleaning": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "overall_churn_rate": float(y.mean()),
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": all_metrics,
        "global_feature_importance": get_global_feature_importance(best_model),
        "notes": [
            "This is a portfolio/demo model trained on the public Telco churn dataset.",
            "Predictions are associations, not proof of causation.",
            "A production deployment would require monitoring, drift checks, security, privacy review, and retraining.",
        ],
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2),
        encoding="utf-8",
    )

    print(f"\nSelected model: {best_name}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")

    return metadata


if __name__ == "__main__":
    train_project()
