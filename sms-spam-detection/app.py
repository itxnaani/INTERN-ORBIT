# app.py
import streamlit as st
import joblib
from src.preprocess import clean_text

# Load model and vectorizer
model = joblib.load('models/model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

st.title("📱 SMS Spam Detector")
st.markdown("**Classify messages as Spam or Legitimate**")

# Initialize session state
if "prediction" not in st.session_state:
    st.session_state.prediction = ""
if "reset_triggered" not in st.session_state:
    st.session_state.reset_triggered = False

# Function to reset the state
def reset():
    st.session_state.prediction = ""
    st.session_state.reset_triggered = True
    st.rerun()

# Text area input
user_input = "" if st.session_state.reset_triggered else st.session_state.get("user_input", "")
user_input = st.text_area("Enter your SMS message:", value=user_input, key="user_input")

# Reset trigger off after render
if st.session_state.reset_triggered:
    st.session_state.reset_triggered = False

# Buttons in columns
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Predict"):
        if user_input.strip():
            cleaned_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vectorized_input)[0]
            label = "🛑 Spam" if prediction == 1 else "✅ Legitimate"
            st.session_state.prediction = f"Prediction: **{label}**"
        else:
            st.warning("Please enter a message.")

with col2:
    if st.button("Reset"):
        reset()

# Show prediction
if st.session_state.prediction:
    st.success(st.session_state.prediction)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Built by <strong>Jasmin Shaik</strong></div>",
    unsafe_allow_html=True
)
