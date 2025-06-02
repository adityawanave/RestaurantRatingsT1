import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load('restaurant_rating_model.pkl')

st.set_page_config(page_title="Restaurant Rating Predictor", layout="wide")

# CSS for compact inputs and styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    body {
        background-color: #f5f7fa;
        font-family: 'Inter', sans-serif;
    }
    .container {
        max-width: 960px;
        margin: 2rem auto 3rem auto;
        background: white;
        padding: 2rem 3rem 3rem 3rem;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }
    h1 {
        font-weight: 700;
        color: #2563eb;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 3rem;
    }
    p.subtitle {
        text-align: center;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 2rem;
        font-size: 1.25rem;
    }
    label[data-baseweb="label"] {
        font-weight: 600;
        color: #334155;
        font-size: 1rem;
    }
    .stNumberInput>div>div>input, .stSelectbox>div>div>div>div {
        border-radius: 8px !important;
        border: 1.8px solid #cbd5e1 !important;
        padding: 8px 12px !important;
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
        font-size: 1.2rem;
        padding: 12px 0;
        width: 100%;
        border-radius: 12px;
        border: none;
        transition: background-color 0.3s ease;
        margin-top: 1.5rem;
    }
    .stButton>button:hover {
        background-color: #1e40af;
        cursor: pointer;
    }
    .result-box {
        margin-top: 2rem;
        background-color: #e0e7ff;
        padding: 1.8rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.6rem;
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
        font-size: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="container">', unsafe_allow_html=True)

    st.markdown("# 🍽️ Restaurant Rating Predictor")
    st.markdown('<p class="subtitle">Enter restaurant details and see your predicted rating with input visualization</p>', unsafe_allow_html=True)

    # Compact form with 5 inputs side by side in 2 rows (columns)
    col1, col2, col3 = st.columns(3)
with col1:
    votes = st.number_input("🗳️ Number of Votes", min_value=0, step=1, format="%d")
with col2:
    cost = st.number_input("💰 Avg Cost for Two (₹)", min_value=0, step=10, format="%d")
with col3:
    price_range = st.selectbox("🏷️ Price Range (1=Low, 4=High)", [1, 2, 3, 4])

col4, col5 = st.columns(2)
with col4:
    online_delivery = st.radio("🚚 Has Online Delivery?", ['Yes', 'No'], horizontal=True)
with col5:
    table_booking = st.radio("📅 Has Table Booking?", ['Yes', 'No'], horizontal=True)


    online_delivery_encoded = 1 if online_delivery == 'Yes' else 0
    table_booking_encoded = 1 if table_booking == 'Yes' else 0

    predict = st.button("Predict Rating")

    if predict:
        input_df = pd.DataFrame([[votes, cost, price_range, online_delivery_encoded, table_booking_encoded]],
                                columns=['Votes', 'Average Cost for two', 'Price range', 'Has Online delivery', 'Has Table booking'])
        prediction = model.predict(input_df)[0]

        # Show prediction result
        st.markdown(f'''
        <div class="result-box">
            <span class="star">⭐</span> Predicted Rating: {prediction:.2f}
        </div>
        ''', unsafe_allow_html=True)

        # Plot a bar chart of inputs for visual context
        st.markdown("### Your Input Summary")
        input_summary = {
            'Votes': votes,
            'Avg Cost (₹)': cost,
            'Price Range': price_range,
            'Online Delivery': online_delivery_encoded * 100,  # convert to % for visibility
            'Table Booking': table_booking_encoded * 100
        }
        fig, ax = plt.subplots(figsize=(8,4))
        bars = ax.bar(input_summary.keys(), input_summary.values(), color="#2563eb", alpha=0.7)
        ax.set_ylim(0, max(input_summary.values()) * 1.2 if max(input_summary.values()) > 0 else 10)
        ax.set_ylabel("Value")
        ax.set_title("Input Feature Values")
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0,3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, color='#334155')

        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)
