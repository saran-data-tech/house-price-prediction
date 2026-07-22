"""
preprocessing.py

This module contains functions for preprocessing
the Ames Housing dataset.
"""

import pandas as pd
from pathlib import Path


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data preprocessing.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """

    df = df.copy()

    # ----------------------------------------------------
    # Columns where missing means feature is absent
    # ----------------------------------------------------
    none_cols = [
        "Pool QC",
        "Alley",
        "Fence",
        "Misc Feature",
        "Fireplace Qu",
        "Garage Type",
        "Garage Finish",
        "Garage Qual",
        "Garage Cond",
        "Bsmt Qual",
        "Bsmt Cond",
        "Bsmt Exposure",
        "BsmtFin Type 1",
        "BsmtFin Type 2",
        "Mas Vnr Type"
    ]

    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # ----------------------------------------------------
    # Numeric columns where missing means 0
    # ----------------------------------------------------
    zero_cols = [
        "Garage Cars",
        "Garage Area",
        "BsmtFin SF 1",
        "BsmtFin SF 2",
        "Bsmt Unf SF",
        "Total Bsmt SF",
        "Bsmt Full Bath",
        "Bsmt Half Bath",
        "Mas Vnr Area"
    ]

    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ----------------------------------------------------
    # Lot Frontage
    # Fill using Neighborhood median
    # ----------------------------------------------------
    if "Lot Frontage" in df.columns:

        neighborhood_median = (
            df.groupby("Neighborhood")["Lot Frontage"]
              .median()
        )

        df["Lot Frontage"] = df["Lot Frontage"].fillna(
            df["Neighborhood"].map(neighborhood_median)
        )

        df["Lot Frontage"] = df["Lot Frontage"].fillna(
            df["Lot Frontage"].median()
        )

    # ----------------------------------------------------
    # Garage Year Built
    # ----------------------------------------------------
    if "Garage Yr Blt" in df.columns:
        df["Garage Yr Blt"] = df["Garage Yr Blt"].fillna(
            df["Year Built"]
        )

    # ----------------------------------------------------
    # Electrical
    # ----------------------------------------------------
    if "Electrical" in df.columns:

        electrical_mode = df["Electrical"].mode()[0]

        df["Electrical"] = df["Electrical"].fillna(
            electrical_mode
        )

    return df


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent

    df = pd.read_csv(BASE_DIR / "data" / "AmesHousing.csv")

    df = preprocess_data(df)

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    print(missing.sort_values(ascending=False))

    print("\nPreprocessing completed successfully!")