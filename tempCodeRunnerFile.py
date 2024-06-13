import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set up the Streamlit interface
st.title('Credit Card Fraud Transaction Detection')

st.write("""
This app predicts whether a credit card transaction is fraudulent or not based on the transaction features.
""")

# Create input fields for the transaction features
features = []
for i in range(1, 30):
    value = st.text_input(f'Feature {i}', '0')
    features.append(float(value))

# When the user clicks the "Predict" button
if st.button('Predict'):
    data = np.array(features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    output = 'Fraudulent' if prediction[0] == 1 else 'Legal'
    st.write(f'Transaction is: {output}')
