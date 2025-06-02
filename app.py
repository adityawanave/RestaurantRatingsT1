import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('restaurant_rating_model.pkl')

# Page config with an emoji icon
st.set_page_config(
    page_title="🍽️ Restaurant Rating Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background: #f9f9f9;
        padding: 2rem 3rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        max-width: 600px;
        margin: auto;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-weight: 800;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        width: 100%;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #e04343;
        cursor: pointer;
    }
    .input-label {
        font-weight: 600;
        color: #333;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)

    st.markdown("# 🍽️ Restaurant Rating Predictor")
    st.markdown("#### Enter restaurant details below to predict the rating:")

    # Use columns for better layout
    votes = st.number_input("Number of Votes", min_value=0, step=1, format="%d")
    cost = st.number_input("Average Cost for Two (₹)", min_value=0, step=10, format="%d")
    price_range = st.selectbox("Price Range", [1, 2, 3, 4], help="1 = Low, 4 = High")
    online_delivery = st.selectbox("Has Online Delivery?", ['Yes', 'No'])
    table_booking = st.selectbox("Has Table Booking?", ['Yes', 'No'])

    # Encode inputs
    online_delivery_encoded = 1 if online_delivery == 'Yes' else 0
    table_booking_encoded = 1 if table_booking == 'Yes' else 0

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Predict Rating"):
        input_df = pd.DataFrame([[votes, cost, price_range, online_delivery_encoded, table_booking_encoded]],
                                columns=['Votes', 'Average Cost for two', 'Price range', 'Has Online delivery', 'Has Table booking'])
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Aggregate Rating: ⭐ {prediction:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)
