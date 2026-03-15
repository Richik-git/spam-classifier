import streamlit as st
import pickle

from src.preprocess import transform_text

# Load model
@st.cache_resource
def load_model():
    with open("models/spam_pipeline.pkl", "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()


st.title("📩 SMS / Email Spam Classifier")

st.write(
    "Enter a message and the model will predict whether it is **Spam** or **Not Spam**."
)

message = st.text_area("Enter your message")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message")

    else:
        cleaned = transform_text(message)

        prediction = model.predict([cleaned])[0]

        if prediction == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")