import pandas as pd
import numpy as np
import bambi as bmb
from .base_model import BaseModel
import arviz as az


def _get_inference_method_chains():
    """
    Bambi is quicker with nuts_blackjax if you have a NVIDIA GPU.

    If not, mcmc is the default inference method which is slower but still
    the best option for CPU.

    This method returns the best possible inference method on a given machine.

    Returns
    -------
    str
        Inference method to use.
    """
    inference_method = "mcmc"
    chains = 4
    try:
        import jax

        if jax.device_count() > 0:
            inference_method = "nuts_blackjax"
            chains = 1  # NUTS is not parallelizable
            print("GPU!!!")
    except ImportError:
        pass

    return inference_method, chains


class RandomEffectsModel(BaseModel):
    def __init__(
        self,
        lags: int = 1,
        country_column: str = "country",
        target_column: str = "inflation",
        exogenous_columns: list = [
            "gdp_growth",
            "interest_rate",
            "unemployment_rate",
            "commodity_CRUDE_PETRO",
            "commodity_iNATGAS",
            "commodity_iAGRICULTURE",
            "commodity_iMETMIN",
            "commodity_iPRECIOUSMET",
        ],
        burn: int = 500,
        draws: int = 5000,
    ):
        """
        Initializes the model.
        """
        self.lags = lags
        self.country_column = country_column
        self.target_column = target_column
        self.exogenous_columns = exogenous_columns
        self.inference_method, self.chains = _get_inference_method_chains()

        # MCMC parameters
        self.burn = burn
        self.draws = draws

        # Build the formula to be used by Bambi
        self.formula = f"{target_column} ~ "

        # Regress on lagged exogenous variables and lagged target variable
        lagged_exog_columns = [
            f"{col}_lag_{i}"
            for i in range(1, lags + 1)
            for col in exogenous_columns + [target_column]
        ]
        self.formula += " + ".join(exogenous_columns + lagged_exog_columns)

    def fit(self, data: pd.DataFrame):
        """
        Fits the model on data.
        """

        # Get country out of the multiindex
        data = data.reset_index(level=self.country_column)

        countries = data[self.country_column].unique()
        if len(countries) <= 1:
            print(
                "WARNING: Only one country in the data in RandomEffectsModel.fit(). Random effects model not applicable."
            )

        assert (
            self.target_column in data.columns
        ), f"Target column {self.target_column} not in data."
        assert all(
            [col in data.columns for col in self.exogenous_columns]
        ), f"Exogenous columns not in data."

        # Create lagged variables
        data = self._create_lagged_variables(data)

        # Drop rows with NaNs due to lagged variables
        data = data.dropna()

        # Fit the model
        self.model = bmb.Model(self.formula, data=data)
        self.results = self.model.fit(
            inference_method=self.inference_method,
            draws=self.draws,
            tune=self.burn,
            chains=self.chains,
        )

    def _create_lagged_variables(self, data: pd.DataFrame):
        """
        Creates lagged variables for the model.

        Takes into account that the data is grouped by country.
        """
        for i in range(1, self.lags + 1):
            for col in self.exogenous_columns + [self.target_column]:
                data[f"{col}_lag_{i}"] = data.groupby(self.country_column)[col].shift(i)

        return data

    def predict(self, data: pd.DataFrame, pointwise_aggregation_method: str = "mean"):
        """
        Predicts the target variable on data.
        """
        # Get country out of the multiindex
        data = data.reset_index(level=self.country_column)

        data = self._create_lagged_variables(data)
        data = data.dropna()

        # Predict using the model that was fit
        predictions = self.model.predict(self.results, data = data, inplace = False)

        if pointwise_aggregation_method == "mean":
            predictions = az.extract(predictions)[f"{self.target_column}_mean"].mean("sample")
        else:
            raise ValueError(f"Unknown pointwise aggregation method: {pointwise_aggregation_method}")

        data["predictions"] = predictions

        return data.reset_index().set_index([self.country_column, "date"])["predictions"].rename("inflation").to_frame()
