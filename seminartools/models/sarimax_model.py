from seminartools.models.base_model import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd


class SARIMAXModel(BaseModel):
    """
    A SARIMAX model for forecasting inflation rates.
    Independent SARIMAX per country.

    Args:
        criterion (str): The criterion to use for model selection. Can be "aic" or "bic".
        exogenous_columns (list[str]): The columns to use as exogenous variables.
        country_column (str): The name of the column containing the country names. Used to split the data into countries.
        inflation_column (str): The name of the column containing the inflation rates.
        Defaults to an empty list, meaning we have a SARIMA model per country (no X).
    """

    def __init__(
        self,
        criterion: str = "aic",
        exogenous_columns: list[str] = [],
        country_column: str = "Country",
        inflation_column: str = "inflation",
        max_p: int = 3,
        max_q: int = 3,
    ):
        self.criterion = criterion
        self.exogenous_columns = exogenous_columns
        self.country_column = country_column
        self.inflation_column = inflation_column
        self.max_p = max_p
        self.max_q = max_q

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
        Auto SARIMAX model

        Args:
            data (pd.DataFrame): data
                Has to have a datetime index, and only contain data from a single country

        Returns:
            SARIMAX: best SARIMAX model
            best_order: best order for SARIMAX model
        """

        best_ic = float("inf")
        best_order = None
        best_model = None

        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
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
                        best_model = results
                elif self.criterion == "bic":
                    if results.bic < best_ic:
                        best_ic = results.bic
                        best_order = (p, q)
                        best_model = results

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
            country_data = data[data[self.country_column] == country]
            country_data = country_data.set_index("yearmonth")
            prediction = self._predict_country(country_data, country)
            prediction_index = country_data.index[-1] + pd.DateOffset(months=3)

            predictions.append({
                "yearmonth": prediction_index,
                "Country": country,
                "inflation": prediction
            })

        return pd.DataFrame(predictions)



    def _predict_country(self, data: pd.DataFrame, country: str) -> pd.DataFrame:
        """
        Predict the inflation rate for the next month for a single country.
        """
        model = self.models[country]
        forecast_result = model.get_forecast(steps=1, exog=data[self.exogenous_columns].iloc[-1].values.reshape(1, -1))
        return forecast_result.predicted_mean[0]
