import streamlit as st
import joblib

model = joblib.load('models/terror_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

st.title("🌍 Global Terrorism Activity Predictor")
st.write("Enter a description of a suspicious activity to predict if it's likely terrorism-related.")

user_input = st.text_area("Description")

if st.button("Predict"):
    if user_input.strip():
        transformed = vectorizer.transform([user_input])
        pred = model.predict(transformed)
        st.success("⚠️ Likely Terrorism Activity" if pred[0] == 1 else "✅ Probably Harmless")
    else:
        st.warning("Please enter some text.")
