# import streamlit as st
# import pandas as pd
# import joblib

# # Load model and scaler
# model = joblib.load('rf_model.pkl')
# scaler = joblib.load('scaler.pkl')

# st.title("🎬 Customer Churn Prediction for Netflix Users")

# st.markdown("""
# This app predicts whether a customer is likely to **churn (cancel subscription)** based on their viewing and account details.
# """)

# # --- Collect only important user inputs ---
# st.header("Enter User Details")

# age = st.number_input('Age', min_value=10, max_value=100, value=25)
# watch_hours = st.number_input('Total Watch Hours (per month)', min_value=0, value=50)
# last_login_days = st.number_input('Days Since Last Login', min_value=0, value=2)
# monthly_fee = st.number_input('Monthly Subscription Fee ($)', min_value=0, value=15)

# # --- Default / Hidden values for other features ---
# gender = 'Female'
# subscription_type = 'Standard'
# region = 'Asia'
# device = 'Mobile'
# payment_method = 'Credit Card'
# number_of_profiles = 2
# avg_watch_time_per_day = 3

# # --- Combine all features in the correct order ---
# X = pd.DataFrame([[
#     age, gender, subscription_type, watch_hours, last_login_days,
#     region, device, monthly_fee, payment_method, number_of_profiles, avg_watch_time_per_day
# ]], columns=[
#     'age', 'gender', 'subscription_type', 'watch_hours', 'last_login_days',
#     'region', 'device', 'monthly_fee', 'payment_method', 'number_of_profiles', 'avg_watch_time_per_day'
# ])

# # --- Encode categorical values like training ---
# X_encoded = pd.get_dummies(X)

# # To ensure feature columns match the model's training data
# expected_features = scaler.feature_names_in_
# for col in expected_features:
#     if col not in X_encoded.columns:
#         X_encoded[col] = 0
# X_encoded = X_encoded[expected_features]

# # --- Scale data ---
# X_scaled = scaler.transform(X_encoded)

# # --- Predict churn ---
# if st.button('Predict Churn'):
#     prediction = model.predict(X_scaled)
#     churn_prob = model.predict_proba(X_scaled)[0][1]

#     if prediction[0] == 1:
#         st.error(f"❌ Customer is **likely to churn**. (Churn probability: {churn_prob:.2f})")
#     else:
#         st.success(f"✅ Customer is **likely to stay subscribed**. (Churn probability: {churn_prob:.2f})")


import streamlit as st
import pandas as pd
import joblib

# --- Page Config ---
st.set_page_config(page_title="Netflix Churn Predictor", page_icon="🎬", layout="centered")

# --- Custom CSS Styling (Dark Netflix Theme) ---
st.markdown("""
    <style>
        /* Background and Text */
        .stApp {
            background-color: #0b0b0b;
            color: #f5f5f5;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Title */
        h1 {
            color: #e50914;
            text-align: center;
            font-weight: 800;
        }

        /* Headers */
        h2, h3 {
            color: #ffffff;
            border-left: 4px solid #e50914;
            padding-left: 10px;
        }

        /* Number Input Boxes */
        input[type=number] {
            background-color: #1c1c1c;
            color: white !important;
            border-radius: 8px;
            border: 1px solid #333;
        }

        /* Buttons */
        div.stButton > button {
            background-color: #e50914;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: bold;
        }

        div.stButton > button:hover {
            background-color: #b20710;
            color: white;
        }

        /* Success & Error boxes */
        .stAlert {
            border-radius: 10px;
        }

    </style>
""", unsafe_allow_html=True)

# --- Load Model and Scaler ---
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- Title and Intro ---
st.title("🎬 Customer Churn Prediction for Netflix Users")
st.markdown("""
This app predicts whether a customer is likely to **churn (cancel subscription)** based on their viewing and account details.
""")

# --- User Inputs ---
st.header("  Enter User Details")

age = st.number_input('Age', min_value=10, max_value=100, value=25)
watch_hours = st.number_input('Total Watch Hours (per month)', min_value=0, value=50)
last_login_days = st.number_input('Days Since Last Login', min_value=0, value=2)
monthly_fee = st.number_input('Monthly Subscription Fee ($)', min_value=0, value=15)

# --- Default Hidden Features ---
gender = 'Female'
subscription_type = 'Standard'
region = 'Asia'
device = 'Mobile'
payment_method = 'Credit Card'
number_of_profiles = 2
avg_watch_time_per_day = 3

# --- Combine Inputs ---
X = pd.DataFrame([[age, gender, subscription_type, watch_hours, last_login_days,
                   region, device, monthly_fee, payment_method, number_of_profiles, avg_watch_time_per_day]],
                 columns=['age', 'gender', 'subscription_type', 'watch_hours', 'last_login_days',
                          'region', 'device', 'monthly_fee', 'payment_method', 'number_of_profiles',
                          'avg_watch_time_per_day'])

# --- One-Hot Encode & Align ---
X_encoded = pd.get_dummies(X)
expected_features = scaler.feature_names_in_
for col in expected_features:
    if col not in X_encoded.columns:
        X_encoded[col] = 0
X_encoded = X_encoded[expected_features]

# --- Scale Data ---
X_scaled = scaler.transform(X_encoded)

# --- Prediction ---
if st.button('Predict Churn'):
    prediction = model.predict(X_scaled)
    churn_prob = model.predict_proba(X_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"❌ Customer is **likely to churn**. (Churn probability: {churn_prob:.2f})")
    else:
        st.success(f"✅ Customer is **likely to stay subscribed**. (Churn probability: {churn_prob:.2f})")

