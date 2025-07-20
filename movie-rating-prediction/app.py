import streamlit as st
import pandas as pd
import joblib

# Load model and vectorizer from .joblib files
model = joblib.load('models/model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

# Streamlit page setup
st.set_page_config(page_title="Movie Rating Predictor 🎬", layout="centered")
st.title("🎥 Movie Rating Predictor")
st.write("Estimate movie ratings based on Genre, Director, and Actors using a trained ML model.")

# Input fields
genre = st.text_input("🎭 Genre", placeholder="e.g., Drama, Comedy")
director = st.text_input("🎬 Director", placeholder="e.g., Rajkumar Hirani")
actors = st.text_input("👥 Actors", placeholder="e.g., Aamir Khan, Kareena Kapoor")

# Predict button
if st.button("Predict Rating"):
    if genre and director and actors:
        # Combine text features as you did during training
        input_text = f"{genre.lower()} {director.lower()} {actors.lower()}"
        
        # Vectorize input
        input_vector = vectorizer.transform([input_text])
        
        # Predict rating
        predicted_rating = model.predict(input_vector)[0]
        
        st.success(f"🌟 Predicted Rating: **{predicted_rating:.2f} / 10**")
    else:
        st.warning("⚠️ Please fill in all the fields to get a prediction.")

st.markdown("---")
st.markdown("Built by Venkatesh Barla")
