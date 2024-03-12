from seminartools.models.base_model import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd


class FourQuarterAverageModel(BaseModel):
    """
    Takes the previous four inflation rates and averages them as a forecast.

    Args:
        country_column (str): The name of the column containing the country names. Used to split the data into countries.
        inflation_column (str): The name of the column containing the inflation rates.
    """

    def __init__(
        self,
        country_column: str = "Country",
        inflation_column: str = "inflation",
    ):
        self.country_column = country_column
        self.inflation_column = inflation_column

    def fit(self, data: pd.DataFrame):
        """
        Fit the model to the data.

        Does nothing, because this model has no parameters.
        """
        pass

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict the inflation rate for the next month.
        """
        # we split into countries
        countries = data[self.country_column].unique()
        predictions = []

        for country in countries:
            country_data = data[data[self.country_column] == country]
            country_data = country_data.set_index("yearmonth")

            # get the last four quarters
            last_four = country_data[self.inflation_column].tail(4)
            # calculate the average
            prediction = last_four.mean()
            # add one quarter to the index

            prediction_index = last_four.index[-1] + pd.DateOffset(months=3)

            prediction = {
                "yearmonth": prediction_index,
                "Country": country,
                "inflation": prediction,
            }

            predictions.append(prediction)

        return pd.DataFrame(predictions)
