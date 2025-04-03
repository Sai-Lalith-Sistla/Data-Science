# IPL Score Prediction Web App (Streamlit)
# -----------------------------------------
# This script provides a web interface for predicting the final score of an IPL match at any given over.
# Users input match details, and the trained TPOT model predicts the expected final score.

# Note:
# - This is the deployment script for the web app.
# - There is a separate notebook that handles dataset preprocessing and model training.




import pandas as pd
import joblib
import streamlit as st
from best_tpot_pipeline import exported_pipeline  # Import the best TPOT model

# Load preprocessed dataset
data = pd.read_csv("preprocessed_ipl_data.csv")

# Function to predict final score
def predict_final_score(over, cumulative_runs, cumulative_wickets, run_rate, batting_team, bowling_team, venue):
    input_data = pd.DataFrame([[over, cumulative_runs, cumulative_wickets, run_rate, batting_team, bowling_team, venue]],
                              columns=['over', 'cumulative_runs', 'cumulative_wickets', 'run_rate', 'batting_team', 'bowling_team', 'venue'])
    predicted_score = exported_pipeline.predict(input_data)[0]
    return round(predicted_score, 2)

# Streamlit UI
st.title("IPL Score Prediction using TPOT")

# User Inputs
over = st.number_input("Over Number", min_value=1, max_value=20, step=1)
cumulative_runs = st.number_input("Cumulative Runs", min_value=0, step=1)
cumulative_wickets = st.number_input("Cumulative Wickets", min_value=0, max_value=10, step=1)
run_rate = st.number_input("Run Rate", min_value=0.0, step=0.1)
batting_team = st.number_input("Batting Team (Encoded)", min_value=0, step=1)
bowling_team = st.number_input("Bowling Team (Encoded)", min_value=0, step=1)
venue = st.number_input("Venue (Encoded)", min_value=0, step=1)

if st.button("Predict Final Score"):
    predicted_score = predict_final_score(over, cumulative_runs, cumulative_wickets, run_rate, batting_team, bowling_team, venue)
    st.success(f"Predicted Final Score: {predicted_score}")
