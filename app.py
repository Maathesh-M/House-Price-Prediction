import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# -------------------------------
# Load the trained model
# -------------------------------
"""with open("model.pickle", "rb") as f:
    model = pickle.load(f)

# Polynomial transformer (same degree used during training)
poly = PolynomialFeatures(degree=2)

# Fit on dummy data so transform() knows the feature structure
dummy = pd.DataFrame({
    "area": [1000],
    "bedrooms": [2],
    "bathrooms": [2]
})
poly.fit(dummy)"""
with open("model.pickle", "rb") as f:
    model = pickle.load(f)

with open("poly.pickle", "rb") as f:
    poly = pickle.load(f)

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Enter the house details below to predict its price.")

# -------------------------------
# User Inputs
# -------------------------------
area = st.number_input(
    "Area (sq.ft)",
    min_value=100,
    max_value=10000,
    value=1400,
    step=50
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms]
    })

    input_poly = poly.transform(input_data)

    prediction = model.predict(input_poly)

    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed using Streamlit and Scikit-learn")