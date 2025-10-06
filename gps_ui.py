import streamlit as st
from streamlit_geolocation import streamlit_geolocation

st.title("Fetch GPS Location of Your Device")

location = streamlit_geolocation()

if location:
    if location["latitude"] and location["longitude"]:
        st.write(f"Latitude: {location['latitude']}")
        st.write(f"Longitude: {location['longitude']}")
        # Display map
        st.map([{'lat': location['latitude'], 'lon': location['longitude']}])
    else:
        st.warning("Press the button to allow geolocation access.")
else:
    st.info("Waiting for location data or user permission...")
