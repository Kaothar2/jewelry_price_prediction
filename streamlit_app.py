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
xgb_pipe = joblib.load("Xgboost_model.pkl")

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

# Define missing columns with default values
input_data = {
    'Category': 'Unknown',  # Placeholder value
    'Target_Gender': selected_gender,  # User input for gender
    'Main_Color': 'Unknown',  # Placeholder value
    'Main_Gem': selected_gemstone,  # User input for gemstone
    'Main_Metal': 'Unknown',  # Placeholder value
    'Brand_ID': 0,  # Placeholder value for numerical column
}

# One-hot encoding for jewelry type
jewelry_types = ["bracelet", "brooch", "earring", "necklace", "pendant", "ring", "souvenir", "stud"]
input_data.update({f"One_hot__x0_jewelry.{j}": 0 for j in jewelry_types})
input_data[f"One_hot__x0_jewelry.{selected_jewelry}"] = 1

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
