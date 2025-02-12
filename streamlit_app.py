import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor

# Load trained model pipeline (this includes imputation, encoding, and the regressor)
xgb_pipe = joblib.load("XGBoost_pipeline.pkl")

# Define one-hot encoding categories
jewelry_types = ["bracelet", "brooch", "earring", "necklace", "pendant", "ring", "souvenir", "stud"]
genders = ["f", "m"]
gemstones = ["rhodolite", "ruby", "sapphire", "sapphire_geothermal", "sitall", "spinel", "topaz", "tourmaline", "turquoise"]

# Streamlit UI
st.title("Jewelry Price Prediction App :gem:")
st.write("Enter jewelry details below to predict the price.")

# User Inputs
selected_jewelry = st.selectbox("Select Jewelry Type", jewelry_types)
selected_gender = st.radio("Select Gender", genders)
selected_gemstone = st.selectbox("Select Gemstone", gemstones)
remainder_x1 = st.number_input("Enter Other Feature (e.g., Weight or Size)", min_value=0.0, step=0.1)

# Convert inputs to one-hot encoding
input_data = {f"One_hot__x0_jewelry.{j}": 0 for j in jewelry_types}
input_data[f"One_hot__x0_jewelry.{selected_jewelry}"] = 1
gender_data = {f"One_hot__x2_{g}": 0 for g in genders}
gender_data[f"One_hot__x2_{selected_gender}"] = 1
gemstone_data = {f"One_hot__x5_{g}": 0 for g in gemstones}
gemstone_data[f"One_hot__x5_{selected_gemstone}"] = 1

# Combine input features
input_data.update(gender_data)
input_data.update(gemstone_data)
input_data["remainder__x1"] = remainder_x1

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Display input DataFrame columns for debugging
st.write("Input DataFrame Columns:", input_df.columns)

# Use the pipeline for prediction
if st.button("Predict Price :moneybag:"):
    try:
        predicted_price = xgb_pipe.predict(input_df)[0]  # Use the entire pipeline for prediction
        st.success(f"Estimated Price: ${predicted_price:.2f}")
    except ValueError as e:
        st.error(f"Error: {e}")
