from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import (
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    get_feature_columns,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_TABLES_DIR = PROJECT_ROOT / "report" / "tables"


def load_processed_splits() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    train_data = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_data = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_data = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    x_train = train_data.drop(columns=["target"])
    y_train = train_data["target"]

    x_val = val_data.drop(columns=["target"])
    y_val = val_data["target"]

    x_test = test_data.drop(columns=["target"])
    y_test = test_data["target"]

    return x_train, x_val, x_test, y_train, y_val, y_test


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features, categorical_features = get_feature_columns(features)

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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def get_cp1_experiments() -> list[dict]:
    return [
        {
            "experiment_group": "baseline_without_feature_engineering",
            "model": "dummy_most_frequent",
            "estimator": DummyClassifier(strategy="most_frequent"),
        },
        {
            "experiment_group": "baseline_without_feature_engineering",
            "model": "logistic_regression",
            "estimator": LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "experiment_group": "with_feature_engineering",
            "model": "decision_tree_depth_6",
            "estimator": DecisionTreeClassifier(
                max_depth=6,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "experiment_group": "with_feature_engineering",
            "model": "decision_tree_depth_10",
            "estimator": DecisionTreeClassifier(
                max_depth=10,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "experiment_group": "with_feature_engineering",
            "model": "random_forest_depth_10",
            "estimator": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "experiment_group": "with_feature_engineering",
            "model": "random_forest_depth_14",
            "estimator": RandomForestClassifier(
                n_estimators=200,
                max_depth=14,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "experiment_group": "with_feature_engineering",
            "model": "extra_trees_depth_14",
            "estimator": ExtraTreesClassifier(
                n_estimators=200,
                max_depth=14,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "experiment_group": "with_feature_engineering",
            "model": "gradient_boosting",
            "estimator": GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
        },
    ]


def evaluate_model(
    experiment_group: str,
    model_name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_val)

    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(x_val)[:, 1]
    else:
        y_score = y_pred

    return {
        "experiment_group": experiment_group,
        "model": model_name,
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_score),
    }


def run_cp1_experiments() -> pd.DataFrame:
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    x_train, x_val, _, y_train, y_val, _ = load_processed_splits()

    results = []

    for experiment in get_cp1_experiments():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                ("model", experiment["estimator"]),
            ]
        )

        metrics = evaluate_model(
            experiment_group=experiment["experiment_group"],
            model_name=experiment["model"],
            pipeline=pipeline,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
        )

        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df.to_csv(REPORT_TABLES_DIR / "cp1_model_results.csv", index=False)

    return results_df


if __name__ == "__main__":
    cp1_results = run_cp1_experiments()
    print(cp1_results)