from seminartools.models.base_model import BaseModel
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from scipy.signal import lfilter
import statsmodels.api as sm


class ARMAXModel(BaseModel):
    """
    A ARMAX model for forecasting inflation rates.
    Independent ARMAX per country.

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
        country_column: str = "country",
        inflation_column: str = "inflation",
        max_p: int = 3,
        max_q: int = 3,
        min_datapts: int = 10,
    ):
        self.criterion = criterion
        self.exogenous_columns = exogenous_columns
        self.country_column = country_column
        self.inflation_column = inflation_column
        self.max_p = max_p
        self.max_q = max_q
        self.min_datapts = min_datapts
        
        # Create dictionaries to store the means and standard deviation for the normalization of the data per country
        self.feature_means = {}
        self.feature_stds = {}
        self.target_mean = {}
        self.target_std = {}


    def fit(self, data: pd.DataFrame):
        """
        Fit the model to the data.
        """
        
        # we split into countries
        data = data.copy()
        countries = data[self.country_column].unique()
        self.models = {}
        self.orders = {}
     
        for country in countries:
            # Normalize the data
            data_country = data[data[self.country_column] == country]
            self.feature_means[country] = data_country[self.exogenous_columns].mean()
            self.feature_stds[country] = data_country[self.exogenous_columns].std()
            self.target_mean[country] = data_country[self.inflation_column].mean()
            self.target_std[country] = data_country[self.inflation_column].std()
            data.loc[data[self.country_column]==country, self.exogenous_columns] = (
                data_country[self.exogenous_columns] - self.feature_means[country]
                ) / self.feature_stds[country]
            data.loc[data[self.country_column]==country, self.inflation_column] = (
                data[self.inflation_column] - self.target_mean[country]
                ) / self.target_std[country]

        #for country in countries:
            country_data = data[data[self.country_column] == country]
            country_data = country_data.set_index("date")
            country_data.index = pd.DatetimeIndex(country_data.index).to_period(
                "Q"
            )  # For quarterly data
            if len(country_data) < self.min_datapts:
                continue # not enough data to fit a model
            model, order = self._fit_model(country_data)
            self.models[country] = model
            self.orders[country] = order

    def _fit_model(self, data: pd.DataFrame):
        """
        Auto ARMAX model

        Args:
            data (pd.DataFrame): data
                Has to have a datetime index, and only contain data from a single country

        Returns:
            ARMAX: best ARMAX model
            best_order: best order for ARMAX model
        """
        data = data.copy()
        best_ic = float("inf")
        best_order = None
        best_model = None

        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                if p == 0 and q == 0:
                    continue

                # quiet
                #model = SARIMAX(
                    #data[self.inflation_column],
                    #order=(p, 0, q),
                    #exog=data[self.exogenous_columns]
                    #if self.exogenous_columns
                    #else None,
                #)
                # ARMAX
                model = ARIMA(
                    data[self.inflation_column],
                    order=(p, 0, q),
                   # enforce_stationarity=False, 
                  #  enforce_invertibility=False,
                    exog=data[self.exogenous_columns]
                    if self.exogenous_columns
                    else None,
                )
                results = model.fit()

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
        data = data.copy()
        countries = data[self.country_column].unique()
        predictions = []
        for country in countries:
            if country not in self.models:
                continue # model was not trained for this country. Make no predictions
            country_data = data[data[self.country_column] == country]

            # Normalize
            country_data[self.exogenous_columns] = (
                country_data[self.exogenous_columns] - self.feature_means[country]
                ) / self.feature_stds[country] 
            
            country_data[self.inflation_column] = (
                country_data[self.inflation_column] - self.target_mean[country]
                ) / self.target_std[country]

            country_data = country_data.set_index("date")
            prediction = self._predict_country(country_data, country)
            prediction_index = country_data.index[-1] + pd.DateOffset(months=3)
            
            predictions.append({
                "date": prediction_index,
                "country": country,
                "inflation": prediction
            })

        return pd.DataFrame(predictions)


    def _predict_country(self, data: pd.DataFrame, country: str, forecast_periods: int = 1) -> pd.DataFrame:
        """
        Predict the inflation rate for the next N periods for a single country using the ARMA part of the model.
    
        Args:
            data (pd.DataFrame): The data for the country, including future values for exogenous variables.
            country (str): The name of the country for which to make the prediction.
            forecast_periods (int): The number of periods ahead to forecast.

        Returns:
            pd.DataFrame: A DataFrame with the predictions.
        """


        # Extract model coefficients
        data = data.copy()
        ar_coefs = self.models[country].arparams
        ma_coefs = self.models[country].maparams
        const = self.models[country].params['const']
        exog_coefs = self.models[country].params[self.exogenous_columns]

        adjusted_series = (data[self.inflation_column] - const).to_numpy()
        epsilons = np.zeros(len(adjusted_series))
        exogSeries = data[self.exogenous_columns].to_numpy()

        for i in range(max(len(ar_coefs), len(ma_coefs)), len(adjusted_series)):
            forecast = 0
            for j in range(len(ar_coefs)):
                forecast += ar_coefs[j] * adjusted_series[i - j - 1]

            for j in range(len(ma_coefs)):
                forecast += ma_coefs[j] * epsilons[i - j - 1]

            for j in range(len(self.exogenous_columns)):
                exogData = exogSeries[:,j]
                forecast += exog_coefs[j] * exogData[i-1]

            epsilons[i] = adjusted_series[i] - forecast
        
        forecast = 0
        for i in range(len(ar_coefs)):
            forecast += ar_coefs[i] * adjusted_series[-i - 1]
        
        for i in range(len(ma_coefs)):
            forecast += ma_coefs[i] * epsilons[-i - 1]

        for i in range(len(self.exogenous_columns)):
            coef = exog_coefs[self.exogenous_columns[i]]
            exog_data = exogSeries[:,i]
            forecast += coef * exog_data[-1]

        pd.Series(epsilons).plot()
        inflation = forecast + const 

        # Denormalize
        denormalizedInflation = inflation * self.target_std[country] + self.target_mean[country]
        return denormalizedInflation