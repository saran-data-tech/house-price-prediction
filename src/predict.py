"""
predict.py

Loads the trained model and predicts house prices.
"""

import joblib
import pandas as pd

from pathlib import Path

from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.model import SELECTED_FEATURES


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"


# --------------------------------------------------
# Load Model
# --------------------------------------------------

def load_model():
    """
    Load the trained pipeline.
    """

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# Predict Price
# --------------------------------------------------

def predict_price(input_data: dict):
    """
    Predict house price from user input.

    Parameters
    ----------
    input_data : dict
        Dictionary containing user inputs.

    Returns
    -------
    float
        Predicted house price.
    """

    # Convert dictionary into DataFrame
    df = pd.DataFrame([input_data])

    # Apply preprocessing
    df = preprocess_data(df)

    # Apply feature engineering
    df = create_features(df)

    # Keep only the required features
    df = df[SELECTED_FEATURES]

    # Load model
    model = load_model()

    # Predict
    prediction = model.predict(df)

    return float(prediction[0])


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    sample_house = {
        "Overall Qual": 7,
        "Gr Liv Area": 1710,
        "Garage Cars": 2,
        "Garage Area": 500,
        "Total Bsmt SF": 856,
        "Full Bath": 2,
        "Year Built": 2003,
        "Neighborhood": "NAmes",

        # Required for feature engineering
        "1st Flr SF": 856,
        "2nd Flr SF": 854,
        "Yr Sold": 2010
    }

    price = predict_price(sample_house)

    print(f"\nPredicted House Price: ${price:,.2f}")