import pandas as pd
import numpy as np
import bami as bmb
from .base_model import BaseModel

def _get_inference_method():
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
    # if gpu available
    inference_method = "mcmc"
    try:
        import jax
        if jax.device_count() > 0:
            inference_method = "nuts_blackjax"
            print("GPU time")
    except ImportError:
        pass

    return inference_method

class RandomEffectsModel(BaseModel):
    def __init__(
        self,
        lags: int = 1,
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
    ):
        """
        Initializes the model.
        """
        self.lags = lags
        self.target_column = target_column
        self.exogenous_columns = exogenous_columns
        self.inference_method = _get_inference_method()

    def fit(self, data: pd.DataFrame):
        """
        Fits the model on data.
        """

        countries = data["Country"].unique()
        if len(countries) <= 1:
            print("WARNING: Only one country in the data in RandomEffectsModel.fit(). Random effects model not applicable.")

        assert (
            self.target_column in data.columns
        ), f"Target column {self.target_column} not in data."
        assert all(
            [col in data.columns for col in self.exogenous_columns]
        ), f"Exogenous columns not in data."

