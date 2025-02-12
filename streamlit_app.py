import joblib
import pandas as pd
import streamlit as st

# Load the pre-trained model pipeline (this includes imputation, encoding, and the regressor)
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

# Display input DataFrame for debugging
st.write("Input DataFrame:", input_df)

# Make predictions
if st.button("Predict Price :moneybag:"):
    try:
        # Ensure we are correctly accessing the model pipeline steps
        preprocessor = xgb_pipe.named_steps['preprocessor']  # Assuming the preprocessor is named 'preprocessor'
        
        # Apply the preprocessing steps
        input_transformed = preprocessor.transform(input_df)
        
        # Predict using the entire pipeline
        predicted_price = xgb_pipe.predict(input_transformed)[0]  # Using the pre-trained pipeline
        st.success(f"Estimated Price: ${predicted_price:.2f}")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
