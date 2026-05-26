from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"

SERVICE_COLUMNS = [
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

app = FastAPI(
    title="Airline Satisfaction Prediction API",
    version="1.0.0",
)


class PassengerInput(BaseModel):
    customer_type: str = "Loyal Customer"
    age: int = Field(35, ge=0, le=120)
    type_of_travel: str = "Business travel"
    travel_class: str = "Business"
    flight_distance: int = Field(1000, ge=0)
    seat_comfort: int = Field(4, ge=0, le=5)
    departure_arrival_time_convenient: int = Field(4, ge=0, le=5)
    food_and_drink: int = Field(4, ge=0, le=5)
    gate_location: int = Field(3, ge=0, le=5)
    inflight_wifi_service: int = Field(4, ge=0, le=5)
    inflight_entertainment: int = Field(4, ge=0, le=5)
    online_support: int = Field(4, ge=0, le=5)
    ease_of_online_booking: int = Field(4, ge=0, le=5)
    on_board_service: int = Field(4, ge=0, le=5)
    leg_room_service: int = Field(4, ge=0, le=5)
    baggage_handling: int = Field(4, ge=0, le=5)
    checkin_service: int = Field(4, ge=0, le=5)
    cleanliness: int = Field(4, ge=0, le=5)
    online_boarding: int = Field(4, ge=0, le=5)
    departure_delay_in_minutes: float = Field(0, ge=0)
    arrival_delay_in_minutes: float = Field(0, ge=0)


def load_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Model file not found. Run: python -m src.train_final_model",
        )

    return joblib.load(MODEL_PATH)


def input_to_frame(passenger: PassengerInput, feature_columns: list[str]) -> pd.DataFrame:
    row = (
        passenger.model_dump()
        if hasattr(passenger, "model_dump")
        else passenger.dict()
    )

    row["class"] = row.pop("travel_class")

    row["total_delay"] = (
        row["departure_delay_in_minutes"] + row["arrival_delay_in_minutes"]
    )
    row["has_delay"] = int(row["total_delay"] > 0)
    row["is_long_flight"] = int(row["flight_distance"] > 1500)
    row["mean_service_score"] = sum(row[column] for column in SERVICE_COLUMNS) / len(
        SERVICE_COLUMNS
    )

    for column in feature_columns:
        if column not in row:
            row[column] = 0

    return pd.DataFrame([row])[feature_columns]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
    }


@app.post("/predict")
def predict(passenger: PassengerInput) -> dict:
    artifact = load_artifact()

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = artifact["threshold"]

    features = input_to_frame(passenger, feature_columns)

    satisfied_probability = float(model.predict_proba(features)[0, 1])
    prediction = int(satisfied_probability >= threshold)
    prediction_label = "satisfied" if prediction == 1 else "dissatisfied"

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "satisfied_probability": satisfied_probability,
    }