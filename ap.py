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

# ORS API config
ORS_API_KEY = "5b3ce3597851110001cf6248805552724db09ce9c0cb432eadd12425e379847df7388287d71c0c67"
ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"

# OpenWeather API config (replace with your key)
WEATHER_API_KEY = "03cc921a8043f443724dbbf1eb9d1481"
WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"

# ----------------------------
# 2. Fetch Hospital Data (Mock API)
# ----------------------------
try:
    hospital_data = requests.get("http://127.0.0.1:8000/hospitals").json()["hospitals"]
except:
    st.error("❌ Could not fetch hospital API data. Make sure FastAPI is running.")
    st.stop()

emergency_map = {"Low": 0, "Medium": 1, "High": 2}
for h in hospital_data:
    h["emergency_load_encoded"] = emergency_map[h["emergency_load"]]

# ----------------------------
# 3. Helper Functions
# ----------------------------
def get_route_data_ors(start_lat, start_lon, end_lat, end_lon):
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
    try:
        X_train_cols_df = pd.read_csv("processed/X_train.csv", nrows=0)
        expected_cols = X_train_cols_df.columns.tolist()
        features_df = features_df[expected_cols]
        scaled_features = scaler.transform(features_df)
        return model.predict(scaled_features)[0]
    except Exception as e:
        st.error(f"❌ ML ETA error: {e}")
        return None

def get_weather_data(lat, lon):
    """Fetch real-time weather from OpenWeatherMap."""
    try:
        params = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"}
        response = requests.get(WEATHER_URL, params=params).json()
        temp = response["main"]["temp"]
        humidity = response["main"]["humidity"]
        condition = response["weather"][0]["main"].lower()

        if "rain" in condition:
            weather_encoded = 2
        elif "storm" in condition or "thunder" in condition:
            weather_encoded = 3
        elif "cloud" in condition:
            weather_encoded = 1
        else:
            weather_encoded = 0

        return temp, humidity, weather_encoded, condition
    except Exception as e:
        st.error(f"❌ Weather API error: {e}")
        return 30, 70, 0, "Clear"

def evaluate_hospitals(amb_lat, amb_lon, temperature, humidity, weather_encoded):
    results = []

    ors_etas = []
    capacity_utilizations = []
    icu_beds_all = []
    emergency_all = []

    for hosp in hospital_data:
        distance_km, ors_eta, geometry = get_route_data_ors(amb_lat, amb_lon, hosp["lat"], hosp["lon"])
        if distance_km is None:
            continue

        icu_beds = hosp["icu_beds"]
        emergency_load_encoded = hosp["emergency_load_encoded"]
        capacity_utilization = hosp["capacity_utilization"]

        ors_etas.append(ors_eta)
        capacity_utilizations.append(capacity_utilization)
        icu_beds_all.append(icu_beds)
        emergency_all.append(emergency_load_encoded)

        hour = datetime.now().hour
        day_of_week = datetime.now().weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        input_data = pd.DataFrame([{
            "ambulance_lat": amb_lat,
            "ambulance_lon": amb_lon,
            "hospital_lat": hosp["lat"],
            "hospital_lon": hosp["lon"],
            "distance_km": distance_km,
            "temperature": temperature,
            "humidity": humidity,
            "weather_encoded": weather_encoded,
            "icu_beds": icu_beds,
            "emergency_load_encoded": emergency_load_encoded,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "capacity_utilization": capacity_utilization
        }])

        ml_eta = get_ml_eta(input_data)

        results.append({
            "Hospital": hosp["name"],
            "Distance (km)": distance_km,
            "ORS ETA (min)": ors_eta,
            "ML ETA (min)": round(ml_eta, 2) if ml_eta else None,
            "ICU Beds": icu_beds,
            "Emergency Load": hosp["emergency_load"],
            "Capacity Utilization": capacity_utilization,
            "Geometry": geometry
        })

    min_eta, max_eta = min(ors_etas), max(ors_etas)
    min_cap, max_cap = min(capacity_utilizations), max(capacity_utilizations)
    min_icu, max_icu = min(icu_beds_all), max(icu_beds_all)
    min_em, max_em = min(emergency_all), max(emergency_all)

    final_results = []
    for r in results:
        norm_eta = (r["ORS ETA (min)"] - min_eta) / (max_eta - min_eta) if max_eta != min_eta else 0
        norm_cap = (r["Capacity Utilization"] - min_cap) / (max_cap - min_cap) if max_cap != min_cap else 0
        norm_em = (emergency_map[r["Emergency Load"]] - min_em) / (max_em - min_em) if max_em != min_em else 0
        norm_icu = 1 - ((r["ICU Beds"] - min_icu) / (max_icu - min_icu) if max_icu != min_icu else 0)

        score = (0.4 * norm_eta) + (0.25 * norm_cap) + (0.2 * norm_em) + (0.15 * norm_icu)
        r["Score"] = round(score, 3)
        final_results.append(r)

    final_results.sort(key=lambda x: x["Score"])
    return final_results

def plot_routes_map(results, amb_lat, amb_lon):
    m = folium.Map(location=[amb_lat, amb_lon], zoom_start=12)
    folium.Marker([amb_lat, amb_lon], tooltip="Ambulance", icon=folium.Icon(color="blue")).add_to(m)
    best = results[0]["Hospital"]
    for hosp in results:
        if hosp["Geometry"]:
            coords = polyline.decode(hosp["Geometry"])
            if coords:
                col = "red" if hosp["Hospital"] == best else "gray"
                folium.PolyLine(coords, color=col, weight=5, opacity=0.8).add_to(m)
                folium.Marker(coords[-1], tooltip=f"{hosp['Hospital']} ({hosp['ORS ETA (min)']} min)",
                              icon=folium.Icon(color="green" if hosp["Hospital"] == best else "gray")).add_to(m)
    return m

# ----------------------------
# 4. Streamlit UI
# ----------------------------
st.title("🚑 Smart Ambulance Hospital Comparison (Real-Time)")

st.sidebar.header("Ambulance Location")
amb_lat = st.sidebar.number_input("Latitude", value=20.2961, format="%.6f")
amb_lon = st.sidebar.number_input("Longitude", value=85.8245, format="%.6f")

# Fetch real-time weather
temperature, humidity, weather_encoded, condition = get_weather_data(amb_lat, amb_lon)
st.sidebar.markdown(f"🌡️ **Weather:** {condition.capitalize()} ({temperature}°C, {humidity}% Humidity)")

# Session state
if "results" not in st.session_state:
    st.session_state.results = None
if "ors_error" not in st.session_state:
    st.session_state.ors_error = None

if st.button("Find Best Hospital"):
    try:
        results = evaluate_hospitals(amb_lat, amb_lon, temperature, humidity, weather_encoded)
        st.session_state.results = results
        st.session_state.ors_error = None
    except Exception as e:
        st.session_state.ors_error = str(e)
        st.session_state.results = None

if st.session_state.ors_error:
    st.error(f"❌ ORS Error: {st.session_state.ors_error}")

elif st.session_state.results:
    best = st.session_state.results[0]
    st.success(f"🏥 Best Hospital: **{best['Hospital']}** "
               f"(ORS ETA: {best['ORS ETA (min)']} min, ML ETA: {best['ML ETA (min)']} min)")

    st.subheader("🏆 Hospital Ranking")
    st.table(pd.DataFrame(st.session_state.results).drop(columns=["Geometry"]))

    st.subheader("🗺️ Routes")
    map_obj = plot_routes_map(st.session_state.results, amb_lat, amb_lon)
    st_folium(map_obj, width=700, height=500)
