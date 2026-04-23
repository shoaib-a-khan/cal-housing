# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:07:13 2026

@author: khan180
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Model Comparison")

# Load models
@st.cache_resource
def load_models():
    return {
        "Linear Regression": pickle.load(open("models/lr_model.pkl", "rb")),
        "Polynomial Regression": pickle.load(open("models/pr_model.pkl", "rb")),
        "Decision Tree": pickle.load(open("models/dt_model.pkl", "rb")),
        "Random Forest": pickle.load(open("models/rf_model.pkl", "rb")),
    }

models = load_models()

# Load test data (or reuse dataset split)
@st.cache_data
def load_data():
    return pd.read_csv("data/housing.csv")

df = load_data()

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# -----------------------------
# Evaluate models
# -----------------------------
from sklearn.metrics import r2_score, mean_squared_error

results = []

for name, model in models.items():
    preds = model.predict(X)
    results.append({
        "Model": name,
        "R2 Score": r2_score(y, preds),
        "RMSE": np.sqrt(mean_squared_error(y, preds))
    })

results_df = pd.DataFrame(results)

# -----------------------------
# Display
# -----------------------------
st.subheader("Performance Metrics")
st.dataframe(results_df)

# Bar charts
st.subheader("R² Comparison")
st.bar_chart(results_df.set_index("Model")["R2 Score"])

st.subheader("RMSE Comparison")
st.bar_chart(results_df.set_index("Model")["RMSE"])

# -----------------------------
# Best Model Highlight
# -----------------------------
best_model = results_df.sort_values("R2 Score", ascending=False).iloc[0]

st.success(f"🏆 Best Model: {best_model['Model']}")