from seminartools.models.base_model import BaseModel
import pandas as pd
import numpy as np
import statsmodels.api as sm


class VARModel(BaseModel):
    """
    Forecasts inflation rates using a VAR model.


    Args:
        country_column (str): The name of the column containing the country names. Used to split the data into countries.
        inflation_column (str): The name of the column containing the inflation rates.
        lags (list[int]): The lags to use in the VAR model.
        long_format (bool): 
            Whether the data is in long format. If True, the data is expected to have columns "yearmonth", "Country", and "inflation".
            If False, the data is expected to have an index of "yearmonth" and columns for each country.
    """

    def __init__(
        self,
        country_column: str = "Country",
        inflation_column: str = "inflation",
        lags: list[int] = [1, 2, 3, 4],
        long_format: bool = True,
    ):
        self.country_column = country_column
        self.inflation_column = inflation_column
        self.lags = lags
        self.long_format = long_format

        self.models = {}


    def fit(self, data: pd.DataFrame):
        """
        Fit the model to the data.
        """
        # Step 0: convert into wide format. Index is yearmonth, column is Country, values are `inflation`.
        if self.long_format:
            data_wide = data.pivot(index="yearmonth", columns="Country", values="inflation")
        else:
            data_wide = data

        # Need to store the columns of the wide data for during prediction
        self.data_wide_columns = data_wide.columns

        for col in data_wide.columns:
            X = pd.concat([
                    data_wide[regressor_col].shift(lag)
                    for lag in self.lags
                    for regressor_col in data_wide.columns
                ],
                axis = 1,
                keys = [
                    f"{regressor_col}_lag_{lag}"
                    for lag in self.lags
                    for regressor_col in data_wide.columns
                ]
            )
            X = X.iloc[max(self.lags):]
            model = sm.OLS(data_wide[col].iloc[max(self.lags):], sm.add_constant(X))
            self.models[col] = model.fit(disp = False)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict the inflation rate for the next month.
        """
        if self.long_format:
            data_wide = data.pivot(index="yearmonth", columns="Country", values="inflation")
            data_wide = data_wide[self.data_wide_columns]
        else:
            data_wide = data

        predictions = {}
        for col in data_wide.columns:
            X = pd.concat(
                [
                    # Shift by 1 fewer lag because we are predicting the next period
                    # This is also the reason why we add 3 months to the index later
                    data_wide[regressor_col].shift(lag - 1) 
                    for lag in self.lags
                    for regressor_col in data_wide.columns
                ],
                axis = 1,
                keys = [
                    f"{regressor_col}_lag_{lag}"
                    for lag in self.lags
                    for regressor_col in data_wide.columns
                ]
            )
            X = X.iloc[max(self.lags):]
            predictions[col] = self.models[col].predict(sm.add_constant(X))

        predictions = pd.DataFrame(predictions)
        
        # Convert the predictions back to long format
        if self.long_format:
            predictions = predictions.stack().reset_index()
            predictions.columns = ["yearmonth", "Country", "inflation"]
            # The predictions are one quarter ahead
            predictions["yearmonth"] += pd.DateOffset(months=3)
        else:
            predictions.index += pd.DateOffset(months=3)




        return predictions
