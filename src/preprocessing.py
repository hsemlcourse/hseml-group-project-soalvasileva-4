import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 506

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

REPORT_DIR = PROJECT_ROOT / "report"
REPORT_IMAGES_DIR = REPORT_DIR / "images"
REPORT_TABLES_DIR = REPORT_DIR / "tables"

TARGET_COLUMN = "satisfaction"

TARGET_MAPPING = {
    "dissatisfied": 0,
    "neutral or dissatisfied": 0,
    "satisfied": 1,
}


def normalize_column_name(column: str) -> str:
    column = column.strip().lower()
    column = re.sub(r"[^a-z0-9]+", "_", column)
    column = re.sub(r"_+", "_", column)
    return column.strip("_")


def load_raw_data(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    csv_files = sorted(data_dir.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    data = pd.read_csv(csv_files[0])
    data.columns = [normalize_column_name(column) for column in data.columns]

    return data


def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = data.select_dtypes(exclude=["number"]).columns.tolist()

    for column in numeric_columns:
        if data[column].isna().sum() > 0:
            data[column] = data[column].fillna(data[column].median())

    for column in categorical_columns:
        if data[column].isna().sum() > 0:
            data[column] = data[column].fillna(data[column].mode()[0])

    return data


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop_duplicates().copy()


def encode_target(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["target"] = (
        data[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(TARGET_MAPPING)
    )

    if data["target"].isna().any():
        unknown_values = data.loc[data["target"].isna(), TARGET_COLUMN].unique()
        raise ValueError(f"Unknown target values: {unknown_values}")

    data["target"] = data["target"].astype(int)

    return data


def clip_delay_outliers(data: pd.DataFrame, quantile: float = 0.99) -> pd.DataFrame:
    data = data.copy()

    delay_columns = [
        "departure_delay_in_minutes",
        "arrival_delay_in_minutes",
    ]

    for column in delay_columns:
        if column in data.columns:
            upper_bound = data[column].quantile(quantile)
            data[column] = data[column].clip(upper=upper_bound)

    return data


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["total_delay"] = (
        data["departure_delay_in_minutes"]
        + data["arrival_delay_in_minutes"]
    )

    data["has_delay"] = (data["total_delay"] > 0).astype(int)

    data["is_long_flight"] = (data["flight_distance"] > 1500).astype(int)

    service_columns = [
        "seat_comfort",
        "departure_arrival_time_convenient",
        "food_and_drink",
        "gate_location",
        "inflight_wifi_service",
        "inflight_entertainment",
        "online_support",
        "ease_of_online_booking",
        "on_board_service",
        "leg_room_service",
        "baggage_handling",
        "checkin_service",
        "cleanliness",
        "online_boarding",
    ]

    existing_service_columns = [
        column for column in service_columns if column in data.columns
    ]

    data["mean_service_score"] = data[existing_service_columns].mean(axis=1)

    return data


def prepare_clean_dataset() -> pd.DataFrame:
    data = load_raw_data()
    data = fill_missing_values(data)
    data = remove_duplicates(data)
    data = encode_target(data)
    data = clip_delay_outliers(data)
    data = add_features(data)

    return data


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = data.drop(columns=[TARGET_COLUMN, "target"])
    target = data["target"]

    return features, target


def split_train_val_test(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def save_processed_splits(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_data = x_train.copy()
    train_data["target"] = y_train.values

    val_data = x_val.copy()
    val_data["target"] = y_val.values

    test_data = x_test.copy()
    test_data["target"] = y_test.values

    train_data.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_data.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_data.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)


def get_feature_columns(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number"]).columns.tolist()

    return numeric_features, categorical_features