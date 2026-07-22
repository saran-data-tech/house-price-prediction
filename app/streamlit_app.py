"""
streamlit_app.py

House Price Prediction Web Application
"""

import sys
from pathlib import Path

# ------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ------------------------------------------------------
# Imports
# ------------------------------------------------------

import streamlit as st

from src.predict import predict_price

# ------------------------------------------------------
# Page Configuration
# ------------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ------------------------------------------------------
# Title
# ------------------------------------------------------

st.title("🏠 House Price Prediction")

st.markdown(
    """
Estimate the selling price of a house using a Machine Learning model.
"""
)

st.sidebar.header("House Details")

# ------------------------------------------------------
# User Inputs
# ------------------------------------------------------

overall_qual = st.sidebar.slider(
    "Overall Quality",
    min_value=1,
    max_value=10,
    value=5
)

gr_liv_area = st.sidebar.number_input(
    "Ground Living Area (sq ft)",
    min_value=300,
    max_value=6000,
    value=1500
)

garage_cars = st.sidebar.slider(
    "Garage Capacity",
    min_value=0,
    max_value=5,
    value=2
)

garage_area = st.sidebar.number_input(
    "Garage Area (sq ft)",
    min_value=0,
    max_value=1500,
    value=500
)

total_bsmt_sf = st.sidebar.number_input(
    "Basement Area (sq ft)",
    min_value=0,
    max_value=3000,
    value=800
)

full_bath = st.sidebar.slider(
    "Full Bathrooms",
    min_value=0,
    max_value=5,
    value=2
)

year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1872,
    max_value=2025,
    value=2000
)

neighborhood = st.sidebar.selectbox(
    "Neighborhood",
    [
        "NAmes",
        "CollgCr",
        "OldTown",
        "Edwards",
        "Somerst",
        "NridgHt",
        "Gilbert",
        "Sawyer",
        "NWAmes",
        "BrkSide"
    ]
)

first_floor = st.sidebar.number_input(
    "1st Floor Area (sq ft)",
    min_value=300,
    max_value=3000,
    value=900
)

second_floor = st.sidebar.number_input(
    "2nd Floor Area (sq ft)",
    min_value=0,
    max_value=3000,
    value=600
)

# ------------------------------------------------------
# Prediction Button
# ------------------------------------------------------

if st.button("Predict House Price"):

    input_data = {

        "Overall Qual": overall_qual,

        "Gr Liv Area": gr_liv_area,

        "Garage Cars": garage_cars,

        "Garage Area": garage_area,

        "Total Bsmt SF": total_bsmt_sf,

        "Full Bath": full_bath,

        "Year Built": year_built,

        "Neighborhood": neighborhood,

        "1st Flr SF": first_floor,

        "2nd Flr SF": second_floor,

        "Yr Sold": 2010

    }

    try:

        predicted_price = predict_price(input_data)

        st.success(
            f"🏠 Estimated House Price: ${predicted_price:,.2f}"
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)

