import streamlit as st
import joblib

# Page configuration (UI setup)
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Title
st.title("📰 Fake News Detection App")
st.markdown("### Check whether a news article is REAL or FAKE")

# Text input
user_input = st.text_area("Enter News Text Here:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)
        probability = model.predict_proba(input_vector)

        confidence = max(probability[0]) * 100

        if prediction[0] == 1:
            st.success(f"✅ This News is REAL")
        else:
            st.error(f"❌ This News is FAKE")

        st.write(f"Confidence Score: {confidence:.2f}%")

# Footer
st.markdown("---")
st.markdown("Developed by Faizan 🚀")