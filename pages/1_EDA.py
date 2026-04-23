# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:06:01 2026

@author: khan180
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Exploratory Data Analysis")

@st.cache_data
def load_data():
    return pd.read_csv("data/housing.csv")

df = load_data()

# -----------------------------
# Dataset Overview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.write("Shape:", df.shape)

# -----------------------------
# Feature Distribution
# -----------------------------
st.subheader("Feature Distributions")

feature = st.selectbox("Select Feature", df.columns)

fig, ax = plt.subplots()
sns.histplot(df[feature], kde=True, ax=ax)
st.pyplot(fig)

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# -----------------------------
# Scatter Plot (Target vs Feature)
# -----------------------------
st.subheader("Feature vs Target")

x_feature = st.selectbox("X-axis", df.columns)
y_feature = st.selectbox("Y-axis", df.columns, index=len(df.columns)-1)

fig, ax = plt.subplots()
sns.scatterplot(data=df, x=x_feature, y=y_feature, ax=ax)
st.pyplot(fig)

# -----------------------------
# Geo Visualization (key insight)
# -----------------------------
st.subheader("Housing Prices by Location")

map_df = df[["Latitude", "Longitude", "Population", "MedHouseVal"]].copy()

# Scale population for marker size
map_df["size"] = df["Population"] / 100

# Normalize price for color mapping (0 → 1)
map_df["price_norm"] = (df["MedHouseVal"] - df["MedHouseVal"].min()) / (
    df["MedHouseVal"].max() - df["MedHouseVal"].min()
)

# Approximate "jet" colormap (blue → green → red)
map_df["color"] = map_df["price_norm"].apply(
    lambda x: [
        int(255 * x),          # red increases
        int(255 * (1 - abs(x - 0.5) * 2)),  # green peak in middle
        int(255 * (1 - x)),    # blue decreases
        160
    ]
)

st.map(map_df, latitude="Latitude", longitude="Longitude", size="size", color="color")


import plotly.express as px

st.subheader("Housing Prices (Interactive Map)")

fig = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    color="MedHouseVal",          # color by price
    size="Population",            # size by population
    color_continuous_scale="Jet", # same as your cmap
    size_max=15,
    zoom=5,
    mapbox_style="carto-positron",
    title="California Housing Prices"
)

st.plotly_chart(fig, use_container_width=True)