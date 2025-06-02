import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('restaurant_rating_model.pkl')

st.set_page_config(page_title="Restaurant Rating Predictor", layout="centered")

st.title("🍽️ Restaurant Rating Predictor")
st.markdown("Enter restaurant details below to predict the rating:")

# Input fields
votes = st.number_input("Number of Votes", min_value=0)
cost = st.number_input("Average Cost for Two", min_value=0)
price_range = st.selectbox("Price Range", [1, 2, 3, 4])
online_delivery = st.selectbox("Has Online Delivery?", ['Yes', 'No'])
table_booking = st.selectbox("Has Table Booking?", ['Yes', 'No'])

# Encode inputs
online_delivery_encoded = 1 if online_delivery == 'Yes' else 0
table_booking_encoded = 1 if table_booking == 'Yes' else 0

# Prediction
if st.button("Predict Rating"):
    input_df = pd.DataFrame([[votes, cost, price_range, online_delivery_encoded, table_booking_encoded]],
                            columns=['Votes', 'Average Cost for two', 'Price range', 'Has Online delivery', 'Has Table booking'])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Aggregate Rating: ⭐ {prediction:.2f}")
