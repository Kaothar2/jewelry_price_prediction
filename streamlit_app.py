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
brands = ["brand_1", "brand_2", "brand_3"]  # Replace with actual brands used in training

# Define required columns (must match model training data)
required_columns = ["Main_Color", "Main_Gem", "Main_Metal", "Brand_ID", "Target_Gender", "Category", "Weight"]

# Streamlit UI
st.title("💎 Jewelry Price Prediction App")
st.write("Enter jewelry details below to predict the price.")

# User Inputs
selected_jewelry = st.selectbox("Select Jewelry Type", jewelry_types)
selected_gender = st.radio("Select Gender", genders)
selected_gemstone = st.selectbox("Select Gemstone", gemstones)
selected_color = st.selectbox("Select Main Color", main_colors)
selected_metal = st.selectbox("Select Main Metal", main_metals)
selected_brand = st.selectbox("Select Brand", brands)
weight = st.number_input("Enter Weight (grams)", min_value=0.1, step=0.1)

# Create DataFrame for prediction
input_data = pd.DataFrame([{
    "Category": selected_jewelry,
    "Target_Gender": selected_gender,
    "Main_Gem": selected_gemstone,
    "Main_Color": selected_color,
    "Main_Metal": selected_metal,
    "Brand_ID": selected_brand,
    "Weight": weight
}])

# Ensure all required columns exist
for col in required_columns:
    if col not in input_data.columns:
        input_data[col] = None  # Fill missing columns

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
