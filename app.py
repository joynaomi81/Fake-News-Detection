import streamlit as st
import joblib

# Load the Logistic Regression model and TF-IDF vectorizer
model = joblib.load('logistic_regression_model.pkt')
vectorizer = joblib.load('tfidf_vectorizer (1).pkt')

# Streamlit app title
st.title("Fake News Detector App")

# Instructions
st.write("""
### Enter the news text below to determine if it's Real or Fake.
""")
# Text input for user to provide news content
user_input = st.text_area("Enter the news content:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text for prediction.")
    else:
        # Transform the input using the loaded TF-IDF vectorizer
        transformed_input = vectorizer.transform([user_input])

        # Make prediction using the loaded model
        prediction = model.predict(transformed_input)

        # Display result based on the new logic
        if prediction[0] == 1:
            st.success("This news is REAL.")
        else:
            st.error("This news is FAKE.")
