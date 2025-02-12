import joblib
import requests
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# URL to the raw model file on GitHub
model_url = "https://github.com/your_username/your_repo/raw/main/models/Xgboost_model.pkl"

# Download the model from GitHub
response = requests.get(model_url)
with open("Xgboost_model.pkl", "wb") as f:
    f.write(response.content)

# Load the model with joblib
xgb_pipe = joblib.load("Xgboost_model.pkl")

# Define possible categories as they were seen during training
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

# User input data
input_data = {
    'Category': selected_jewelry,  # From user input
    'Target_Gender': selected_gender,  # From user input
    'Main_Color': 'red',  # Placeholder or default value from training data
    'Main_Gem': selected_gemstone,  # From user input
    'Main_Metal': 'gold',  # Placeholder or default value from training data
    'Brand_ID': 0,  # Placeholder value for numerical column
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Create a pipeline that handles the OneHotEncoder for categorical features
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

# Column transformer for encoding categorical columns (OneHot)
column_transformer = ColumnTransformer(
    transformers=[
        ("one_hot", one_hot_encoder, ['Category', 'Target_Gender', 'Main_Gem', 'Main_Color', 'Main_Metal'])
    ],
    remainder='passthrough'
)

# Display input DataFrame for debugging
st.write("Input DataFrame:", input_df)

# Make predictions
if st.button("Predict Price :moneybag:"):
    try:
        # Apply transformations and predict using the pipeline
        input_transformed = column_transformer.fit_transform(input_df)
        predicted_price = xgb_pipe.predict(input_transformed)[0]
        st.success(f"Estimated Price: ${predicted_price:.2f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
