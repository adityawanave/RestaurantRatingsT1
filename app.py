import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('restaurant_rating_model.pkl')

st.set_page_config(
    page_title="🍽️ Restaurant Rating Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&family=Roboto&display=swap');

    body {
        background: linear-gradient(135deg, #f8b500, #fceabb);
        font-family: 'Roboto', sans-serif;
    }

    .main-card {
        background: white;
        max-width: 550px;
        margin: 3rem auto 5rem auto;
        padding: 2.5rem 3rem 3rem 3rem;
        border-radius: 25px;
        box-shadow: 0 20px 50px rgba(245, 166, 35, 0.3);
        transition: all 0.3s ease-in-out;
    }
    .main-card:hover {
        box-shadow: 0 30px 70px rgba(245, 166, 35, 0.5);
        transform: translateY(-5px);
    }

    h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #f59e0b;
        text-align: center;
        font-size: 3.2rem;
        margin-bottom: 0.25rem;
        letter-spacing: 2px;
    }
    h4 {
        text-align: center;
        color: #6b5b00;
        margin-bottom: 2.5rem;
        font-weight: 600;
    }

    label[data-baseweb="label"] {
        font-weight: 600;
        color: #a16207;
        margin-top: 1.2rem;
    }

    .stNumberInput>div>div>input {
        border-radius: 12px !important;
        border: 2px solid #f59e0b !important;
        padding: 12px 15px !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #553c03 !important;
        background: #fff8dc !important;
        transition: border-color 0.3s ease;
    }
    .stNumberInput>div>div>input:focus {
        border-color: #fbbf24 !important;
        outline: none !important;
        box-shadow: 0 0 10px #fbbf24aa;
    }

    .stSelectbox>div>div>div>div {
        border-radius: 12px !important;
        border: 2px solid #f59e0b !important;
        padding: 12px 15px !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #553c03 !important;
        background: #fff8dc !important;
        transition: border-color 0.3s ease;
    }
    .stSelectbox>div>div>div>div:hover {
        border-color: #fbbf24 !important;
        box-shadow: 0 0 10px #fbbf24aa;
        cursor: pointer;
    }

    .stButton>button {
        background: linear-gradient(45deg, #fbbf24, #f59e0b);
        color: white;
        font-size: 1.25rem;
        font-weight: 700;
        padding: 15px 0;
        width: 100%;
        border-radius: 25px;
        border: none;
        box-shadow: 0 8px 15px rgba(245, 158, 11, 0.4);
        transition: all 0.3s ease;
        letter-spacing: 1.1px;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #f59e0b, #d97706);
        box-shadow: 0 12px 25px rgba(215, 120, 3, 0.6);
        transform: translateY(-3px);
        cursor: pointer;
    }

    .result-box {
        margin-top: 2.5rem;
        background: #fffbea;
        padding: 1.8rem 2.5rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.7rem;
        font-weight: 700;
        color: #b45309;
        box-shadow: 0 0 15px 4px #fbbf24aa;
        letter-spacing: 2px;
    }
    .star {
        color: #f59e0b;
        font-size: 2.5rem;
        margin-right: 0.4rem;
        animation: glow 1.5s ease-in-out infinite alternate;
    }
    @keyframes glow {
        0% { text-shadow: 0 0 5px #f59e0b; }
        100% { text-shadow: 0 0 20px #fbbf24, 0 0 30px #fbbf24;}
    }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.markdown("# 🍽️ Restaurant Rating Predictor")
    st.markdown("#### Enter restaurant details below to predict the rating:")

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
            <span class="star">⭐</span> Predicted Aggregate Rating: {prediction:.2f}
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
