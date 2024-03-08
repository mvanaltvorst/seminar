from .base_model import BaseModel
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd


class ARMAModel(BaseModel):
    def __init__(self, criterion: str = "aic"):
        self.criterion = criterion

    def fit(self, data: pd.DataFrame):
        # we split into countries
        countries = data.index.get_level_values("Country").unique()
        self.models = {}
        self.orders = {}

        for country in countries:
            country_data = data.loc[country]
            country_data = country_data.set_index("yearmonth")
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
            ARIMA: best ARIMA model
            best_order: best order for ARIMA model
        """

        best_aic = float("inf")
        best_order = None
        best_model = None

        for p in range(5):
            for d in range(1):
                for q in range(5):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        model_fit = model.fit()
                        aic = model_fit.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            best_model = model_fit
                    except:
                        print(f"Error with order {(p, d, q)}")
                        continue

        return best_model, best_order

    def predict(self, data: pd.DataFrame) -> pd.Series:
        # we split into countries
        countries = data.index.get_level_values("Country").unique()
        predictions = []

        for country in countries:
            country_data = data.loc[country]
            country_data = country_data.set_index("yearmonth")
            prediction = self._predict_country(country_data, country)
            predictions.append(prediction)

        return pd.concat(predictions)
