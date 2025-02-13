import joblib
import pandas as pd
import streamlit as st

# Load the model from the local directory
@st.cache_resource
def load_model():
    model_path = "Xgboost_model.pkl"  # Ensure this file exists in the same directory as your script
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Make sure Xgboost_model.pkl is in the correct directory.")
        return None

# Load the model
xgb_pipe = load_model()

# Define input categories
jewelry_types = ["bracelet", "brooch", "earring", "necklace", "pendant", "ring", "souvenir", "stud"]
genders = ["f", "m"]
gemstones = ["rhodolite", "ruby", "sapphire", "sapphire_geothermal", "sitall", "spinel", "topaz", "tourmaline", "turquoise"]

# Streamlit UI
st.title("Jewelry Price Prediction App 💎")
st.write("Enter jewelry details below to predict the price.")

# User Inputs
selected_jewelry = st.selectbox("Select Jewelry Type", jewelry_types)
selected_gender = st.radio("Select Gender", genders)
selected_gemstone = st.selectbox("Select Gemstone", gemstones)
remainder_x1 = st.number_input("Enter Other Feature (e.g., Weight or Size)", min_value=0.0, step=0.1)

# Convert inputs to one-hot encoding format (ensure correct column names)
input_data = {
    f"One_hot__x0_jewelry.{selected_jewelry}": 1,
    f"One_hot__x0_gender.{selected_gender}": 1,
    f"One_hot__x0_gemstone.{selected_gemstone}": 1,
    "remainder_x1": remainder_x1
}

# Create DataFrame
input_df = pd.DataFrame([input_data])

# Ensure all necessary columns exist
if xgb_pipe:
    try:
        predicted_price = xgb_pipe.predict(input_df)[0]
        st.success(f"Predicted Price: ${predicted_price:.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
