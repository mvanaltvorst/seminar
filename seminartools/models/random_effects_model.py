import pandas as pd
import numpy as np
from .base_model import BaseModel

class RandomEffectsModel(BaseModel):
    def __init__(self):
        pass

    def fit(self, data: pd.DataFrame):
        """
        Fits the model on data.
        """

        countries = data["Country"].unique()

        