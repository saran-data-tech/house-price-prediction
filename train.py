"""
train.py

Main training script for the House Price Prediction project.
"""

import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.model import (
    build_pipeline,
    SELECTED_FEATURES
)


def main():

    # --------------------------------------------------
    # Project Paths
    # --------------------------------------------------

    BASE_DIR = Path(__file__).resolve().parent

    DATA_PATH = BASE_DIR / "data" / "AmesHousing.csv"

    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    MODEL_PATH = MODEL_DIR / "house_price_model.pkl"

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    print("=" * 50)
    print("Loading Dataset...")
    print("=" * 50)

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset Shape : {df.shape}")

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    print("\nPreprocessing Data...")

    df = preprocess_data(df)

    # --------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------

    print("Creating Engineered Features...")

    df = create_features(df)

    # --------------------------------------------------
    # Feature Selection
    # --------------------------------------------------

    X = df[SELECTED_FEATURES]

    y = df["SalePrice"]

    print(f"\nTraining Features : {X.shape[1]}")

    print(f"Selected Features :")

    for feature in SELECTED_FEATURES:
        print(f"   • {feature}")

    # --------------------------------------------------
    # Split Dataset
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # --------------------------------------------------
    # Build Pipeline
    # --------------------------------------------------

    pipeline = build_pipeline(X_train)

    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------

    print("\nTraining Gradient Boosting Model...")

    pipeline.fit(X_train, y_train)

    print("Training Completed.")

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    predictions = pipeline.predict(X_test)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n")
    print("=" * 50)
    print("Model Performance")
    print("=" * 50)

    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R²   : {r2:.4f}")

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print("\nModel Saved Successfully.")

    print(MODEL_PATH)


if __name__ == "__main__":
    main()