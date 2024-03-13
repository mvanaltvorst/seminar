from seminartools.models.base_model import BaseModel
import pandas as pd

class UCSVModel(BaseModel):
    """
    The unobserved component stochastic volatility model to forecast inflation.

    Args:
        ... 
    """

    def __init__(self, country_column: str = "Country", inflation_column: str = "inflation"):
        self.country_column = country_column
        self.inflation_column = inflation_column

    def fit(self, data: pd.DataFrame):
        """
        Fit the model to the data.
        
        """
        pass
