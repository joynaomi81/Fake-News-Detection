import streamlit as st
import joblib

# Load the Logistic Regression model and TF-IDF vectorizer
model = joblib.load('logistic_regression_model (1).pkl')
vectorizer = joblib.load('tfidf_vectorizer (7).pkl')

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

        # Display result
        if prediction[0] == 1:
            st.error("This news is  FAKE.")
        else:
            st.success("This news is REAL.")
