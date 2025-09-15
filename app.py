# app.py
import polyline
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
import folium
from streamlit_folium import st_folium

# ----------------------------
# 1. Load ML model & scaler
# ----------------------------
try:
    model = joblib.load("models/eta_prediction_xgboost_model.pkl")
    scaler = joblib.load("processed/scaler.pkl")
except:
    st.error("❌ Model or scaler not found. Please train and preprocess first.")
    st.stop()

# ----------------------------
# 2. Hospital list
# ----------------------------
hospitals = [
    {"name": "Apollo", "lat": 19.0750, "lon": 72.9050},
    {"name": "JJ Hospital", "lat": 18.9647, "lon": 72.8311},
    {"name": "Fortis", "lat": 19.0840, "lon": 72.8990},
]

# ORS API config
ORS_API_KEY = "5b3ce3597851110001cf6248805552724db09ce9c0cb432eadd12425e379847df7388287d71c0c67"
ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"


# ----------------------------
# 3. Helper Functions
# ----------------------------
def get_route_data_ors(start_lat, start_lon, end_lat, end_lon):
    """Fetch distance, duration & geometry from ORS."""
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    coords = [[start_lon, start_lat], [end_lon, end_lat]]
    body = {"coordinates": coords}

    try:
        response = requests.post(ORS_URL, json=body, headers=headers)
        data = response.json()
        summary = data["routes"][0]["summary"]
        geometry = data["routes"][0]["geometry"]

        distance_km = round(summary["distance"] / 1000, 2)
        duration_min = round(summary["duration"] / 60, 2)
        return distance_km, duration_min, geometry
    except Exception as e:
        st.error(f"❌ ORS error: {e}")
        return None, None, None


def get_ml_eta(features_df):
    """Predict ETA using ML model."""
    try:
        X_train_cols_df = pd.read_csv("processed/X_train.csv", nrows=0)
        expected_cols = X_train_cols_df.columns.tolist()
        features_df = features_df[expected_cols]
        scaled_features = scaler.transform(features_df)
        return model.predict(scaled_features)[0]
    except Exception as e:
        st.error(f"❌ ML ETA error: {e}")
        return None


def evaluate_hospitals(amb_lat, amb_lon, temperature, humidity, weather_encoded, icu_beds, emergency_load_encoded):
    results = []

    ors_etas = []
    capacity_utilizations = []
    icu_beds_all_hospitals = []
    emergency_loads_all_hospitals = []

    for i,hosp in enumerate(hospitals):
        # ORS ETA
        distance_km, ors_eta, geometry = get_route_data_ors(amb_lat, amb_lon, hosp["lat"], hosp["lon"])
        if distance_km is None:
            continue

        # Simulate capacity
        total_beds = np.random.randint(200, 800)
        beds_occupied = np.random.randint(50, total_beds)
        capacity_utilization = beds_occupied / total_beds
        hour = datetime.now().hour
        day_of_week = datetime.now().weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        ors_etas.append(ors_eta)
        capacity_utilizations.append(capacity_utilization)
        icu_beds_all_hospitals.append(icu_beds_list[i])
        emergency_loads_all_hospitals.append(emergency_load_list[i])
        
        

        # ML ETA
        input_data = pd.DataFrame([{
            "ambulance_lat": amb_lat,
            "ambulance_lon": amb_lon,
            "hospital_lat": hosp["lat"],
            "hospital_lon": hosp["lon"],
            "distance_km": distance_km,
            "temperature": temperature,
            "humidity": humidity,
            "weather_encoded": weather_encoded,
            "icu_beds": icu_beds_list[i],
            "emergency_load_encoded": emergency_load_list[i],
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "capacity_utilization": capacity_utilization
        }])
        
        ml_eta=get_ml_eta(input_data)

        

        norm_eta = 0
        norm_capacity = 0
        norm_emergency = 0
        norm_icu_beds = 0

        score = 0

        results.append({
            "Hospital": hosp["name"],
            "Distance (km)": round(distance_km, 2),
            "ORS ETA (min)": ors_eta,
            "ML ETA (min)": round(ml_eta, 2) if ml_eta else None,
            "Humidity (%)": humidity,
            "Weather Code": weather_encoded,
            "ICU Beds": icu_beds_list[i],
            "Emergency Load": emergency_load_list[i],
            "Capacity Utilization": round(capacity_utilization, 2),
            "Score": round(score, 3),
            "Geometry": geometry
        })
    min_ors_eta = min(ors_etas)
    max_ors_eta = max(ors_etas)
    min_capacity = min(capacity_utilizations)
    max_capacity = max(capacity_utilizations)
    min_icu = min(icu_beds_all_hospitals)
    max_icu = max(icu_beds_all_hospitals)
    min_emergency = min(emergency_loads_all_hospitals)
    max_emergency = max(emergency_loads_all_hospitals)
  
    final_results = []
    for result in results:
        ors_eta = result["ORS ETA (min)"]
        capacity_utilization = result["Capacity Utilization"]
        icu_beds = result["ICU Beds"]
        emergency_load_encoded = result["Emergency Load"]

        norm_eta = (ors_eta - min_ors_eta) / (max_ors_eta - min_ors_eta) if max_ors_eta != min_ors_eta else 0
        norm_capacity = (capacity_utilization - min_capacity) / (max_capacity - min_capacity) if max_capacity != min_capacity else 0
        norm_emergency = (emergency_load_encoded - min_emergency) / (max_emergency - min_emergency) if max_emergency != min_emergency else 0
        norm_icu_beds = 1 - ((icu_beds - min_icu) / (max_icu - min_icu) if max_icu != min_icu else 0)  # More beds → better score

        score = (
            0.4 * norm_eta +
            0.25 * norm_capacity +
            0.2 * norm_emergency +
            0.15 * norm_icu_beds
        )
        result["Score"] = round(score, 3)
        final_results.append(result)
        
    final_results.sort(key=lambda x: x["Score"])
    return final_results
        
    


def plot_routes_map(results, amb_lat, amb_lon):
    m = folium.Map(location=[amb_lat, amb_lon], zoom_start=12)
    folium.Marker([amb_lat, amb_lon], tooltip="Ambulance", icon=folium.Icon(color="blue")).add_to(m)
    best = results[0]["Hospital"]
    for hosp in results:
        if hosp["Geometry"]:
            coords = polyline.decode(hosp["Geometry"])
            if coords:  # ✅ Only proceed if coords is not empty
                col = "red" if hosp["Hospital"] == best else "gray"
                folium.PolyLine(coords, color=col, weight=5, opacity=0.8).add_to(m)
                folium.Marker(coords[-1], tooltip=f"{hosp['Hospital']} ({hosp['ORS ETA (min)']} min)",
                              icon=folium.Icon(color="green" if hosp["Hospital"] == best else "gray")).add_to(m)
    return m

# ----------------------------
# 4. Streamlit UI
# ----------------------------
st.title("🚑 Smart Ambulance Hospital Comparison")

st.sidebar.header("Ambulance Location")
amb_lat = st.sidebar.number_input("Latitude", value=19.0760, format="%.6f")
amb_lon = st.sidebar.number_input("Longitude", value=72.8777, format="%.6f")

st.sidebar.header("Weather")
temperature = st.sidebar.slider("Temperature (°C)", 10, 45, 32)
humidity = st.sidebar.slider("Humidity (%)", 10, 100, 80)
weather_options = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Stormy": 3}
selected_weather = st.sidebar.selectbox("Weather", list(weather_options.keys()))
weather_encoded = weather_options[selected_weather]

st.sidebar.header("Hospital Load Settings")

hospital_names = ["Hospital A", "Hospital B", "Hospital C"]

# Lists to store per-hospital values
icu_beds_list = []
emergency_load_list = []

emergency_options = {"Low": 0, "Medium": 1, "High": 2}

for hospital in hospital_names:
    st.sidebar.subheader(hospital)

    beds = st.sidebar.slider(
        f"{hospital} ICU Beds",
        min_value=5, max_value=100, value=30
    )
    icu_beds_list.append(beds)

    selected_emergency = st.sidebar.selectbox(
        f"{hospital} Emergency Load",
        list(emergency_options.keys())
    )
    emergency_load_list.append(emergency_options[selected_emergency])
if "results" not in st.session_state:
    st.session_state.results = None

# Button
if st.button("Find Best Hospital"):
    st.session_state.results = evaluate_hospitals(
        amb_lat, amb_lon, temperature, humidity, weather_encoded,
        icu_beds_list, emergency_load_list
    )

# Display results if available in session state
if st.session_state.results:
    results = st.session_state.results
    best = results[0]
    st.success(f"🏥 Best Hospital: **{best['Hospital']}** "
               f"(ORS ETA: {best['ORS ETA (min)']} min, ML ETA: {best['ML ETA (min)']} min)")

    st.subheader("🏆 Hospital Ranking")
    st.table(pd.DataFrame(results).drop(columns=["Geometry"]))

    st.subheader("🗺️ Routes")
    map_obj = plot_routes_map(results, amb_lat, amb_lon)
    st_folium(map_obj, width=700, height=500)

