from .base_model import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd


class SARIMAXModel(BaseModel):
    """
    An ARMAX model for forecasting inflation rates.
    Independent ARMAX per country.

    Args:
        criterion (str): The criterion to use for model selection. Can be "aic" or "bic".
        exogenous_columns (list[str]): The columns to use as exogenous variables.
        country_column (str): The name of the column containing the country names. Used to split the data into countries.
        inflation_column (str): The name of the column containing the inflation rates.
        Defaults to an empty list, meaning we have an ARMA model per country.
    """

    def __init__(
        self,
        criterion: str = "aic",
        exogenous_columns: list[str] = [],
        country_column: str = "Country",
        inflation_column: str = "inflation",
    ):
        self.criterion = criterion
        self.exogenous_columns = exogenous_columns
        self.country_column = country_column
        self.inflation_column = inflation_column

    def fit(self, data: pd.DataFrame):
        """
        Fit the model to the data.
        """
        # we split into countries
        countries = data[self.country_column].unique()
        self.models = {}
        self.orders = {}

        for country in countries:
            country_data = data[data[self.country_column] == country]
            country_data = country_data.set_index("yearmonth")
            country_data.index = pd.DatetimeIndex(country_data.index).to_period(
                "Q"
            )  # For quarterly data
            model, order = self._fit_model(country_data)
            self.models[country] = model
            self.orders[country] = order

    def _fit_model(self, data: pd.DataFrame):
        """
        Auto ARIMA model

        Args:
            data (pd.DataFrame): data
                Has to have a datetime index, and only contain data from a single country

        Returns:
            ARIMAX: best ARIMA model
            best_order: best order for ARIMA model
        """

        best_ic = float("inf")
        best_order = None
        best_model = None

        for p in range(6):
            for q in range(6):
                if p == 0 and q == 0:
                    continue

                # quiet
                model = SARIMAX(
                    data[self.inflation_column],
                    seasonal_order=(p, 0, q, 4),
                    exog=data[self.exogenous_columns]
                    if self.exogenous_columns
                    else None,
                )
                results = model.fit(disp=False)
                if self.criterion == "aic":
                    if results.aic < best_ic:
                        best_ic = results.aic
                        best_order = (p, q)
                        best_model = model
                elif self.criterion == "bic":
                    if results.bic < best_ic:
                        best_ic = results.bic
                        best_order = (p, q)
                        best_model = model

                # except Exception as e:
                #     print(f"Error with order {(p, d, q)}: {e}")
                #     continue

        return best_model, best_order

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict the inflation rate for the next month.
        """
        # we split into countries
        countries = data[self.country_column].unique()
        predictions = []

        for country in countries:
            country_data = data.loc[country]
            country_data = country_data.set_index("yearmonth")
            prediction = self._predict_country(country_data, country)
            predictions.append(prediction)

        return pd.concat(predictions)
