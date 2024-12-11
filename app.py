import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

# Load pre-trained model and vectorizer (ensure these files are saved during training)
@st.cache(allow_output_mutation=True)
def load_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

model, vectorizer = load_model()

# Define a prediction function
def predict_sentiment(text):
    processed_text = vectorizer.transform([text])  # Transform input using the vectorizer
    prediction = model.predict(processed_text)[0]  # Predict sentiment
    return "Positive" if prediction == 1 else "Negative"

# Streamlit Web App
st.title("Sentiment Analysis Web App")
st.subheader("Analyze the sentiment of your text (Positive or Negative)")

# User Input
user_input = st.text_area("Enter your text here:")

if st.button("Analyze"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.write(f"The sentiment of the entered text is: **{sentiment}**")
    else:
        st.write("Please enter some text to analyze.")

