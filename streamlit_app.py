import streamlit as st
import pickle
import pandas as pd

# Load the trained XGBoost model & encoders
@st.cache_resource
def load_model():
    with open("Xgboost_model.pkl", "rb") as file:
        model, label_encoders = pickle.load(file)
    return model, label_encoders

model, label_encoders = load_model()

# Streamlit UI
st.title("Jewelry Price Prediction App 💎")
st.write("Enter jewelry details to predict the price.")

# User Inputs
category = st.selectbox("Select Jewelry Category", label_encoders["Category"].classes_)
target_gender = st.selectbox("Select Target Gender", label_encoders["Target_Gender"].classes_)
main_color = st.selectbox("Select Main Color", label_encoders["Main_Color"].classes_)
main_gem = st.selectbox("Select Main Gemstone", label_encoders["Main_Gem"].classes_)
main_metal = st.selectbox("Select Main Metal", label_encoders["Main_Metal"].classes_)

# Predict Button
if st.button("Predict Price"):
    # Convert user inputs using Label Encoders
    input_data = pd.DataFrame([[category, target_gender, main_color, main_gem, main_metal]],
                              columns=["Category", "Target_Gender", "Main_Color", "Main_Gem", "Main_Metal"])
    
    for col in input_data.columns:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Make prediction
    predicted_price = model.predict(input_data)[0]
    
    st.success(f"Estimated Price: **${predicted_price:.2f}** 💰")
