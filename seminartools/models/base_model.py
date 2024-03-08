from abc import ABC, abstractmethod
import pandas as pd


def Model(ABC):
    @abstractmethod
    def fit(self, data: pd.DataFrame):
        ...

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        ...
