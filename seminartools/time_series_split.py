from typing import Iterator
import pandas as pd
from abc import ABC, abstractmethod

class TimeSeriesSplit(ABC):
    """
    A class to split a dataframe into multiple time series splits
    based on dates.

    This is applied on the dataframe with features already calculated
    to ensure we don't lose sequence information.
    """
    @abstractmethod
    def split(self, data: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Splits the data into multiple time series splits.
        """
        ...

class ExpandingWindowSplit(TimeSeriesSplit):
    """
    Splits the data based on an expanding window.
    """
    def __init__(self, start_date: str, num_splits: int = 5, date_column: str = "date"):
        self.start_date = pd.to_datetime(start_date)
        self.num_splits = num_splits
        self.date_column = date_column


    def split(self, data: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Splits the data into multiple time series splits.
        """
        end_date = data[self.date_column].max()
        split_size = (end_date - self.start_date) / self.num_splits

        for i in range(self.num_splits):
            split_start = self.start_date + i * split_size
            split_end = split_start + split_size

            train = data[data[self.date_column] < split_start]
            test = data[(data[self.date_column] >= split_start) & (data[self.date_column] <= split_end)]

            yield train, test