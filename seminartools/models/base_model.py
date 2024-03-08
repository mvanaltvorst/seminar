from abc import ABC, abstractmethod
import pandas as pd


class BaseModel(ABC):
    """
    Base class for models.
    """

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
