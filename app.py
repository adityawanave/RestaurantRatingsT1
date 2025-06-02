import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('restaurant_rating_model.pkl')

st.set_page_config(page_title="Restaurant Rating Predictor", layout="centered")

# CSS for clean minimal styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    body {
        background-color: #f5f7fa;
        font-family: 'Inter', sans-serif;
    }
    .container {
        max-width: 480px;
        margin: 3rem auto 4rem auto;
        background: white;
        padding: 2.5rem 2.5rem 3rem 2.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }
    h1 {
        font-weight: 700;
        color: #2563eb;
        text-align: center;
        margin-bottom: 0.25rem;
        font-size: 2.8rem;
    }
    p.subtitle {
        text-align: center;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    label[data-baseweb="label"] {
        font-weight: 600;
        color: #334155;
        margin-top: 1.3rem;
        font-size: 1rem;
    }
    .stNumberInput>div>div>input, .stSelectbox>div>div>div>div {
        border-radius: 8px !important;
        border: 1.8px solid #cbd5e1 !important;
        padding: 10px 12px !important;
        font-size: 1rem !important;
        color: #334155 !important;
        background: #f9fafb !important;
        transition: border-color 0.2s ease-in-out;
    }
    .stNumberInput>div>div>input:focus, .stSelectbox>div>div>div>div:hover {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 5px #3b82f6aa;
        outline: none !important;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 12px 0;
        width: 100%;
        border-radius: 12px;
        border: none;
        transition: background-color 0.3s ease;
        margin-top: 2rem;
    }
    .stButton>button:hover {
        background-color: #1e40af;
        cursor: pointer;
    }
    .result-box {
        margin-top: 2rem;
        background-color: #e0e7ff;
        padding: 1.6rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e40af;
        letter-spacing: 1px;
        user-select: none;
        box-shadow: 0 0 10px #2563eb44;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0.5rem;
    }
    .star {
        color: #2563eb;
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="container">', unsafe_allow_html=True)

    st.markdown("# 🍽️ Restaurant Rating Predictor")
    st.markdown('<p class="subtitle">Enter restaurant details below to predict the rating</p>', unsafe_allow_html=True)

    votes = st.number_input("Number of Votes", min_value=0, step=1, format="%d")
    cost = st.number_input("Average Cost for Two (₹)", min_value=0, step=10, format="%d")
    price_range = st.selectbox("Price Range", [1, 2, 3, 4], help="1 = Low, 4 = High")
    online_delivery = st.selectbox("Has Online Delivery?", ['Yes', 'No'])
    table_booking = st.selectbox("Has Table Booking?", ['Yes', 'No'])

    online_delivery_encoded = 1 if online_delivery == 'Yes' else 0
    table_booking_encoded = 1 if table_booking == 'Yes' else 0

    if st.button("Predict Rating"):
        input_df = pd.DataFrame([[votes, cost, price_range, online_delivery_encoded, table_booking_encoded]],
                                columns=['Votes', 'Average Cost for two', 'Price range', 'Has Online delivery', 'Has Table booking'])
        prediction = model.predict(input_df)[0]

        st.markdown(f'''
        <div class="result-box">
            <span class="star">⭐</span> Predicted Rating: {prediction:.2f}
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
