import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "price_regressor.pkl"
MONTH_ORDER = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
ITEM_OPTIONS = [
    "big onions imported",
    "brinjals",
    "cabbage",
    "carrot",
    "coconut large",
    "coconut small",
    "dried chillies",
    "dried chillies imported",
    "green beans",
    "green chillies",
    "kolikuttu",
    "lime",
    "papaw",
    "potatoes imported",
    "potatoes n'eliya",
    "potatoes welimada",
    "pumpkin",
    "raw (red)",
    "raw (white)",
    "red dhal",
    "red onion imported",
    "red onions sinnan",
    "red onions vedalan",
    "samba 1",
    "samba 2",
    "seeni",
    "snake gourd",
    "tomatoes",
    "ambul",
    "imported potato",
    "imported samba",
    "nadu 1",
    "nadu 2",
]
MONTH_OPTIONS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    try:
        with open(MODEL_PATH, "rb") as model_file:
            return pickle.load(model_file)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The saved model could not be loaded because the required sklearn dependency is missing or incompatible. "
            "Install scikit-learn==1.6.1, pandas, and streamlit from requirements.txt."
        ) from exc


def get_model_choices(model):
    feature_names = model.feature_names_in_.tolist()

    item_map = {
        name.replace("item_", ""): name
        for name in feature_names
        if name.startswith("item_") and name != "item_"
    }
    month_map = {
        name.replace("month_", ""): name
        for name in feature_names
        if name.startswith("month_") and name != "month_num"
    }

    item_labels = ITEM_OPTIONS if ITEM_OPTIONS else sorted(item_map.keys(), key=lambda x: x.lower())
    month_labels = MONTH_OPTIONS if MONTH_OPTIONS else sorted(month_map.keys(), key=lambda x: MONTH_ORDER.get(x.lower(), 99))

    return item_labels, month_labels, item_map, month_map


def build_prediction_input(selected_item, selected_month, year, temperature, rain, lp95, lp92, lad, lsd, lk, lik, item_map, month_map):
    model = load_model()
    feature_names = model.feature_names_in_.tolist()
    row = {name: 0 for name in feature_names}

    row["year"] = year
    row["temperature"] = temperature
    row["rain"] = rain
    row["LP95"] = lp95
    row["LP92"] = lp92
    row["LAD"] = lad
    row["LSD"] = lsd
    row["LK"] = lk
    row["LIK"] = lik
    row["month_num"] = MONTH_ORDER.get(selected_month.lower(), 1)

    item_field = item_map.get(selected_item)
    month_field = month_map.get(selected_month)

    if item_field and item_field in row:
        row[item_field] = 1
    if month_field and month_field in row:
        row[month_field] = 1

    return pd.DataFrame([row], columns=feature_names)


def main():
    st.set_page_config(page_title="Farm Produce Price Prediction", layout="wide")
    st.title("💰 Farm Produce Price Prediction")
    st.caption("Select a crop or produce item and estimate its expected market price.")

    try:
        model = load_model()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    item_options, month_options, item_map, month_map = get_model_choices(model)

    with st.form("price_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            item = st.selectbox("Crop / Produce", item_options, index=0)
            month = st.selectbox("Month", month_options, index=0)
            year = st.number_input("Year", min_value=2020, max_value=2035, value=2024, step=1)
            temperature = st.number_input("Temperature (°C)", value=28.0, step=0.5, format="%.1f")
            rain = st.number_input("Rainfall (mm)", value=120.0, step=1.0)

        with col2:
            lp95 = st.number_input("LP95", value=120.0, step=0.5, format="%.2f")
            lp92 = st.number_input("LP92", value=110.0, step=0.5, format="%.2f")
            lad = st.number_input("LAD", value=95.0, step=0.5, format="%.2f")
            lsd = st.number_input("LSD", value=88.0, step=0.5, format="%.2f")
            lk = st.number_input("LK", value=90.0, step=0.5, format="%.2f")
            lik = st.number_input("LIK", value=92.0, step=0.5, format="%.2f")

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        input_df = build_prediction_input(
            selected_item=item,
            selected_month=month,
            year=year,
            temperature=temperature,
            rain=rain,
            lp95=lp95,
            lp92=lp92,
            lad=lad,
            lsd=lsd,
            lk=lk,
            lik=lik,
            item_map=item_map,
            month_map=month_map,
        )

        prediction = model.predict(input_df)[0]
        st.success(f"Predicted market price for {item}: LKR {prediction:,.2f}")
        st.metric("Estimated market price", f"LKR {prediction:,.2f}")

        st.subheader("Selected inputs")
        st.write({
            "Crop": item,
            "Month": month,
            "Year": year,
            "Temperature (°C)": temperature,
            "Rainfall (mm)": rain,
        })

        st.subheader("Model input row")
        st.dataframe(input_df, use_container_width=True)

    st.caption("Model file: price_regressor.pkl")


if __name__ == "__main__":
    main()
