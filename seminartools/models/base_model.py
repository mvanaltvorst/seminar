from abc import ABC, abstractmethod
import pandas as pd


class BaseModel(ABC):
    """
    Base class for models.

    REQUIRES_ANTE_FULL_FIT: bool
        Whether the model requires fitting on the entire dataset before predicting.
        Some models benefit from fitting on the entire dataset before predicting, such as the particle filter.
        The particle filter stores intermediate states, which allows us to very quickly evaluate the model on new data.
        However, some models do not benefit from this, such as the ARIMA model.
        The default is False, meaning that the model does not require fitting on the entire dataset before predicting.
    """
    REQUIRES_ANTE_FULL_FIT = False

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """
        Fit the model to the data.
        """
        ...

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict one step ahead given the data.
        """
        ...
