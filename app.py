import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model and feature names
with open('best_model_with_features.pkl', 'rb') as file:
    linear_model, feature_names = pickle.load(file)

# Define function to preprocess user inputs for regression model
def preprocess_input_for_regression(followers, comments, sentiment):
    # Set default values for all features
    default_values = {
        'Followers': 1000,  # Typical follower count
        'Comments': 50,     # Typical comment count
        'Sentiment': 0      # Neutral sentiment
    }

    # Update default values with user inputs
    input_data = {
        'Followers': followers,
        'Comments': comments,
        'Sentiment': sentiment
    }

    default_values.update(input_data)

    # Create DataFrame and ensure it has the same feature order as during training
    input_df = pd.DataFrame([default_values])[feature_names]
    return input_df

# Streamlit app
st.title('Social Media Likes Prediction System')

st.header('Input Features')
followers = st.number_input('Followers', min_value=0, max_value=10000, value=500)
comments = st.number_input('Comments', min_value=0, max_value=600, value=100)
sentiment = st.selectbox('Sentiment', ['Positive', 'Neutral', 'Negative'])
sentiment = 1 if sentiment == 'Positive' else -1 if sentiment == 'Negative' else 0

# Add a button to trigger the prediction
if st.button('Predict'):
    # Preprocess the input
    input_df = preprocess_input_for_regression(followers, comments, sentiment)

    # Predict Likes using the Linear Regression model
    likes_predicted = linear_model.predict(input_df)

    st.header('Prediction')
    st.write(f'Estimated Likes: {likes_predicted[0]:.2f}')
