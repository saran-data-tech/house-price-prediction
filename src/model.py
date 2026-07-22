"""
model.py

Defines the machine learning pipeline used for
House Price Prediction.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.ensemble import GradientBoostingRegressor


# ----------------------------------------------------
# Features used by the final model
# ----------------------------------------------------

SELECTED_FEATURES = [

    "Overall Qual",

    "Gr Liv Area",

    "Garage Cars",

    "Garage Area",

    "Total Bsmt SF",

    "Full Bath",

    "Year Built",

    "Neighborhood",

    "TotalSF",

    "HouseAge"

]


def build_pipeline(X):
    """
    Build the preprocessing and modelling pipeline.

    Parameters
    ----------
    X : pandas.DataFrame

    Returns
    -------
    sklearn.pipeline.Pipeline
    """

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "numeric",
                numeric_transformer,
                numeric_features
            ),

            (
                "categorical",
                categorical_transformer,
                categorical_features
            )

        ]
    )

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                GradientBoostingRegressor(
                    random_state=42,
                    learning_rate=0.1,
                    max_depth=3,
                    n_estimators=200
                )
            )

        ]

    )

    return pipeline