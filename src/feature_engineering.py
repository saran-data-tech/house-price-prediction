"""
feature_engineering.py

Feature engineering module for the Ames Housing dataset.
Works for both training data and user input.
"""

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    # --------------------------------------------------
    # Default values for optional columns
    # --------------------------------------------------

    default_columns = {

        "1st Flr SF": 0,
        "2nd Flr SF": 0,
        "Total Bsmt SF": 0,

        "Full Bath": 0,
        "Half Bath": 0,
        "Bsmt Full Bath": 0,
        "Bsmt Half Bath": 0,

        "Year Built": 0,
        "Year Remod/Add": None,
        "Yr Sold": 2010,

        "Open Porch SF": 0,
        "Enclosed Porch": 0,
        "3Ssn Porch": 0,
        "Screen Porch": 0,

        "Garage Area": 0,
        "Fireplaces": 0
    }

    for column, value in default_columns.items():

        if column not in df.columns:

            if column == "Year Remod/Add":

                if "Year Built" in df.columns:
                    df[column] = df["Year Built"]
                else:
                    df[column] = 0

            else:
                df[column] = value

    # --------------------------------------------------
    # Total Square Footage
    # --------------------------------------------------

    df["TotalSF"] = (

        df["1st Flr SF"]

        + df["2nd Flr SF"]

        + df["Total Bsmt SF"]

    )

    # --------------------------------------------------
    # Total Bathrooms
    # --------------------------------------------------

    df["TotalBath"] = (

        df["Full Bath"]

        + (0.5 * df["Half Bath"])

        + df["Bsmt Full Bath"]

        + (0.5 * df["Bsmt Half Bath"])

    )

    # --------------------------------------------------
    # House Age
    # --------------------------------------------------

    df["HouseAge"] = (

        df["Yr Sold"]

        - df["Year Built"]

    )

    # --------------------------------------------------
    # Remodel Age
    # --------------------------------------------------

    df["RemodAge"] = (

        df["Yr Sold"]

        - df["Year Remod/Add"]

    )

    # --------------------------------------------------
    # Is Remodeled
    # --------------------------------------------------

    df["IsRemodeled"] = (

        df["Year Built"]

        != df["Year Remod/Add"]

    ).astype(int)

    # --------------------------------------------------
    # Total Porch Area
    # --------------------------------------------------

    df["TotalPorchSF"] = (

        df["Open Porch SF"]

        + df["Enclosed Porch"]

        + df["3Ssn Porch"]

        + df["Screen Porch"]

    )

    # --------------------------------------------------
    # Binary Features
    # --------------------------------------------------

    df["HasGarage"] = (

        df["Garage Area"] > 0

    ).astype(int)

    df["HasBasement"] = (

        df["Total Bsmt SF"] > 0

    ).astype(int)

    df["HasFireplace"] = (

        df["Fireplaces"] > 0

    ).astype(int)

    return df


if __name__ == "__main__":

    print("Feature Engineering Module Loaded Successfully!")