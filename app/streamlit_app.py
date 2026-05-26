from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

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


@st.cache_resource
def load_artifact() -> dict:
    return joblib.load(MODEL_PATH)


def build_features(row: dict, feature_columns: list[str]) -> pd.DataFrame:
    row = row.copy()

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


st.set_page_config(
    page_title="Airline Satisfaction Prediction",
    page_icon="✈️",
)

st.title("Предсказание удовлетворённости авиапассажира")

if not MODEL_PATH.exists():
    st.error("Модель не найдена. Сначала выполните: python -m src.train_final_model")
    st.stop()

artifact = load_artifact()
model = artifact["model"]
feature_columns = artifact["feature_columns"]
threshold = artifact["threshold"]

customer_type = st.selectbox(
    "Тип клиента",
    ["Loyal Customer", "disloyal Customer"],
)
age = st.slider("Возраст", 0, 100, 35)
type_of_travel = st.selectbox(
    "Тип поездки",
    ["Business travel", "Personal Travel"],
)
travel_class = st.selectbox(
    "Класс обслуживания",
    ["Business", "Eco", "Eco Plus"],
)
flight_distance = st.number_input("Дистанция перелёта", min_value=0, value=1000)

st.subheader("Оценки сервиса")

seat_comfort = st.slider("Seat comfort", 0, 5, 4)
departure_arrival_time_convenient = st.slider(
    "Departure/Arrival time convenient",
    0,
    5,
    4,
)
food_and_drink = st.slider("Food and drink", 0, 5, 4)
gate_location = st.slider("Gate location", 0, 5, 3)
inflight_wifi_service = st.slider("Inflight wifi service", 0, 5, 4)
inflight_entertainment = st.slider("Inflight entertainment", 0, 5, 4)
online_support = st.slider("Online support", 0, 5, 4)
ease_of_online_booking = st.slider("Ease of online booking", 0, 5, 4)
on_board_service = st.slider("On-board service", 0, 5, 4)
leg_room_service = st.slider("Leg room service", 0, 5, 4)
baggage_handling = st.slider("Baggage handling", 0, 5, 4)
checkin_service = st.slider("Checkin service", 0, 5, 4)
cleanliness = st.slider("Cleanliness", 0, 5, 4)
online_boarding = st.slider("Online boarding", 0, 5, 4)

st.subheader("Задержки")

departure_delay_in_minutes = st.number_input(
    "Departure delay in minutes",
    min_value=0.0,
    value=0.0,
)
arrival_delay_in_minutes = st.number_input(
    "Arrival delay in minutes",
    min_value=0.0,
    value=0.0,
)

row = {
    "customer_type": customer_type,
    "age": age,
    "type_of_travel": type_of_travel,
    "class": travel_class,
    "flight_distance": flight_distance,
    "seat_comfort": seat_comfort,
    "departure_arrival_time_convenient": departure_arrival_time_convenient,
    "food_and_drink": food_and_drink,
    "gate_location": gate_location,
    "inflight_wifi_service": inflight_wifi_service,
    "inflight_entertainment": inflight_entertainment,
    "online_support": online_support,
    "ease_of_online_booking": ease_of_online_booking,
    "on_board_service": on_board_service,
    "leg_room_service": leg_room_service,
    "baggage_handling": baggage_handling,
    "checkin_service": checkin_service,
    "cleanliness": cleanliness,
    "online_boarding": online_boarding,
    "departure_delay_in_minutes": departure_delay_in_minutes,
    "arrival_delay_in_minutes": arrival_delay_in_minutes,
}

if st.button("Сделать предсказание"):
    features = build_features(row, feature_columns)
    satisfied_probability = float(model.predict_proba(features)[0, 1])
    prediction = int(satisfied_probability >= threshold)
    prediction_label = "satisfied" if prediction == 1 else "dissatisfied"

    st.metric(
        "Вероятность удовлетворённости",
        f"{satisfied_probability:.1%}",
    )

    if prediction_label == "satisfied":
        st.success(f"Предсказанный класс: {prediction_label}")
    else:
        st.warning(f"Предсказанный класс: {prediction_label}")