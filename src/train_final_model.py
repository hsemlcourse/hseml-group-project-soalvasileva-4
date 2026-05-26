from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 506

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_TABLES_DIR = PROJECT_ROOT / "report" / "tables"

MODEL_PATH = MODELS_DIR / "final_model.joblib"


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_data = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_data = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_data = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    train_val_data = pd.concat([train_data, val_data], ignore_index=True)

    x_train_val = train_val_data.drop(columns=["target"])
    y_train_val = train_val_data["target"]

    x_test = test_data.drop(columns=["target"])
    y_test = test_data["target"]

    return x_train_val, y_train_val, x_test, y_test


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = GradientBoostingClassifier(
        learning_rate=0.07,
        max_depth=5,
        min_samples_leaf=1,
        min_samples_split=4,
        n_estimators=250,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict:
    predictions = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)[:, 1]

    return {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "roc_auc": roc_auc_score(target, probabilities),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    x_train_val, y_train_val, x_test, y_test = load_data()

    pipeline = build_pipeline(x_train_val)
    pipeline.fit(x_train_val, y_train_val)

    test_metrics = evaluate_model(pipeline, x_test, y_test)

    artifact = {
        "model": pipeline,
        "feature_columns": x_train_val.columns.tolist(),
        "model_name": "gradient_boosting_tuned",
        "positive_class": "satisfied",
        "threshold": 0.5,
        "test_metrics": test_metrics,
    }

    joblib.dump(artifact, MODEL_PATH)

    pd.DataFrame([test_metrics]).to_csv(
        REPORT_TABLES_DIR / "cp3_final_model_test_results.csv",
        index=False,
    )

    print(f"Model saved to: {MODEL_PATH}")
    print(test_metrics)


if __name__ == "__main__":
    main()