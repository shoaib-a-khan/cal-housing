# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:06:45 2026

@author: khan180
"""

import streamlit as st
import pandas as pd
import pickle

st.title("Housing Price Prediction")

@st.cache_resource
def load_models():
    return {
        "Linear Regression": pickle.load(open("models/lr_model.pkl", "rb")),
        "Polynomial Regression": pickle.load(open("models/pr_model.pkl", "rb")),
        "Decision Tree": pickle.load(open("models/dt_model.pkl", "rb")),
        "Random Forest": pickle.load(open("models/rf_model.pkl", "rb")),
    }

models = load_models()

# Sidebar Inputs
st.sidebar.header("Input Features")

def user_input():
    return pd.DataFrame([{
        "MedInc": st.sidebar.slider("Median Income", 0.0, 15.0, 3.0),
        "HouseAge": st.sidebar.slider("House Age", 1, 50, 20),
        "AveRooms": st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.0),
        "AveBedrms": st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0),
        "Population": st.sidebar.slider("Population", 100, 5000, 1000),
        "AveOccup": st.sidebar.slider("Occupancy", 1.0, 10.0, 3.0),
        "Latitude": st.sidebar.slider("Latitude", 32.0, 42.0, 36.0),
        "Longitude": st.sidebar.slider("Longitude", -125.0, -114.0, -120.0),
    }])

input_df = user_input()

model_name = st.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

prediction = model.predict(input_df)[0]

st.metric("Predicted Value", f"${prediction*100000:,.0f}")

# Compare toggle
if st.checkbox("Compare all models"):
    results = {name: m.predict(input_df)[0] for name, m in models.items()}
    st.bar_chart(pd.DataFrame.from_dict(results, orient="index"))