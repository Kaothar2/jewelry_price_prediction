import joblib
import pandas as pd
import streamlit as st

# Load model
@st.cache_resource
def load_model():
    model_path = "Xgboost_model.pkl"  # Ensure this file is in the same directory as your script
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Ensure Xgboost_model.pkl is in the correct directory.")
        return None

# Load the pipeline
xgb_pipe = load_model()

# Define expected categories based on training
jewelry_types = ["bracelet", "brooch", "earring", "necklace", "pendant", "ring", "souvenir", "stud"]
genders = ["f", "m"]
gemstones = ["rhodolite", "ruby", "sapphire", "sapphire_geothermal", "sitall", "spinel", "topaz", "tourmaline", "turquoise"]
main_colors = ["red", "blue", "green", "black", "white"]
main_metals = ["gold", "silver", "platinum"]

# Define required columns (must match model training data)
required_columns = ["Category", "Target_Gender", "Main_Gem", "Main_Color", "Main_Metal"]

# Streamlit UI
st.title("💎 Jewelry Price Prediction App")
st.write("Enter jewelry details below to predict the price.")

# User Inputs
selected_jewelry = st.selectbox("Select Jewelry Type", jewelry_types)
selected_gender = st.radio("Select Gender", genders)
selected_gemstone = st.selectbox("Select Gemstone", gemstones)
selected_color = st.selectbox("Select Main Color", main_colors)
selected_metal = st.selectbox("Select Main Metal", main_metals)

# Create DataFrame for prediction (Ensure only the necessary columns are included)
input_data = pd.DataFrame([{
    "Category": selected_jewelry,
    "Target_Gender": selected_gender,
    "Main_Gem": selected_gemstone,
    "Main_Color": selected_color,
    "Main_Metal": selected_metal
}])

# Ensure the columns are in the correct order (matching what the model expects)
input_data = input_data[required_columns]

# Display the input data
st.write("### Input Data:")
st.dataframe(input_data)

# Make prediction
if st.button("Predict Price"):
    if xgb_pipe:
        try:
            predicted_price = xgb_pipe.predict(input_data)[0]  # Ensure input format is correct
            st.success(f"💰 Predicted Jewelry Price: ${predicted_price:.2f}")
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
    else:
        st.error("Model not loaded properly.")
