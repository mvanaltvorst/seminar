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
            This is useful for the VAR model because this is an intermediate step for the PCA VAR model, and for the PCA VAR
            model is it useful that we can pass wide format data to this VAR model.
    """

    def __init__(
        self,
        country_column: str = "Country",
        date_column: str = "yearmonth",
        inflation_column: list[str] = ["inflation"],
        country_exogenous_columns: list[str] = [],
        global_exogenous_columns: list[str] = [],
        lags: list[int] = [1, 2, 3, 4],
        country_exog_lags: list[int] = [1],
        global_exog_lags: list[int] = [1],
        long_format: bool = True
    ):
        self.country_column = country_column
        self.date_column = date_column
        self.inflation_column = inflation_column
        self.country_exogenous_columns = country_exogenous_columns
        self.global_exogenous_columns = global_exogenous_columns
        self.lags = lags
        self.country_exog_lags = country_exog_lags
        self.global_exog_lags = global_exog_lags
        self.long_format = long_format


        self.models = {}


    def fit(self, data: pd.DataFrame):
        """
        Fit the model to the data.
        """
        # Step 0: convert into wide format. Index is yearmonth, column is Country, values are `inflation`.
        if self.long_format:
            data_wide = data.pivot(index=self.date_column, columns=self.country_column, values=self.inflation_column)
            data_wide_country_exog = data.pivot(index=self.date_column, columns=self.country_column, values=self.country_exogenous_columns)
            data_wide_country_exog.columns = data_wide_country_exog.columns.map('_'.join)
            data_wide_global_exog = data.pivot(index=self.date_column, columns=self.country_column, values=self.global_exogenous_columns)
            top_level_indices = data_wide_global_exog.columns.get_level_values(0).unique() # Global exogenous variables are the same for all countries, so we can just take one column
            columns_to_keep = []
            for top_index in top_level_indices:
                sub_columns = data_wide_global_exog[top_index].columns
                if len(sub_columns) > 0:
                    columns_to_keep.append((top_index, sub_columns[0]))
            data_wide_global_exog = data_wide_global_exog.loc[:, columns_to_keep]
            data_wide_global_exog.columns = data_wide_global_exog.columns.map('_'.join)
            data_wide_global_exog.columns = self.global_exogenous_columns
        else:
            data_wide = data[self.inflation_column]
            data_wide_country_exog = data[self.country_exogenous_columns]
            data_wide_global_exog = data[self.global_exogenous_columns]

        # data_wide = data_wide.fillna(0)
        # data_wide_country_exog = data_wide_country_exog.fillna(0)
        # data_wide_global_exog = data_wide_global_exog.fillna(0)

        
        # Need to store the columns of the wide data for during prediction
        self.data_wide_columns = data_wide.columns

        # Do country by country OLS
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

            if not data_wide_country_exog.empty:
                X_country_exog = pd.concat([
                    data_wide_country_exog[regressor_col].shift(lag)
                    for lag in self.country_exog_lags
                    for regressor_col in data_wide_country_exog.columns
                    ],
                    axis=1,
                    keys=[
                        f"{regressor_col}_lag_{lag}"
                        for lag in self.country_exog_lags
                        for regressor_col in data_wide_country_exog.columns
                    ]
                )
            else:
                X_country_exog = pd.DataFrame()

            if not data_wide_global_exog.empty:
                X_global_exog = pd.concat([
                    data_wide_global_exog[regressor_col].shift(lag)
                    for lag in self.global_exog_lags
                    for regressor_col in data_wide_global_exog.columns
                    ],
                    axis=1,
                    keys=[
                        f"{regressor_col}_lag_{lag}"
                        for lag in self.global_exog_lags
                        for regressor_col in data_wide_global_exog.columns
                    ]
                )
            else:
                X_global_exog = pd.DataFrame()

            to_concatenate = [X, X_country_exog, X_global_exog]
            to_concatenate = [df for df in to_concatenate if not df.empty]
            all_X = pd.concat(to_concatenate, axis=1)
            all_X = all_X.iloc[max(self.lags):]
            model = sm.OLS(data_wide[col].iloc[max(self.lags):], sm.add_constant(all_X))
            self.models[col] = model.fit(disp = False)

            # X = X.iloc[max(self.lags):]
            # model = sm.OLS(data_wide[col].iloc[max(self.lags):], sm.add_constant(X))
            # self.models[col] = model.fit(disp = False)
            

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict the inflation rate for the next month.
        """
        
        if self.long_format:
            data_wide = data.pivot(index=self.date_column, columns=self.country_column, values=self.inflation_column)
            data_wide_country_exog = data.pivot(index=self.date_column, columns=self.country_column, values=self.country_exogenous_columns)
            data_wide_country_exog.columns = data_wide_country_exog.columns.map('_'.join)
            data_wide_global_exog = data.pivot(index=self.date_column, columns=self.country_column, values=self.global_exogenous_columns)
            top_level_indices = data_wide_global_exog.columns.get_level_values(0).unique() # Global exogenous variables are the same for all countries, so we can just take one column
            columns_to_keep = []
            for top_index in top_level_indices:
                sub_columns = data_wide_global_exog[top_index].columns
                if len(sub_columns) > 0:
                    columns_to_keep.append((top_index, sub_columns[0]))
            data_wide_global_exog = data_wide_global_exog.loc[:, columns_to_keep]
            data_wide_global_exog.columns = data_wide_global_exog.columns.map('_'.join)
            data_wide_global_exog.columns = self.global_exogenous_columns
        else:
            data_wide = data[self.inflation_column]
            data_wide_country_exog = data[self.country_exogenous_columns]
            data_wide_global_exog = data[self.global_exogenous_columns]

        predictions = {}
        for col in data_wide.columns:
            X = pd.concat([
                    data_wide[regressor_col].shift(lag-1)
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

            if not data_wide_country_exog.empty:
                X_country_exog = pd.concat([
                    data_wide_country_exog[regressor_col].shift(lag-1)
                    for lag in self.country_exog_lags
                    for regressor_col in data_wide_country_exog.columns
                    ],
                    axis=1,
                    keys=[
                        f"{regressor_col}_lag_{lag}"
                        for lag in self.country_exog_lags
                        for regressor_col in data_wide_country_exog.columns
                    ]
                )
            else:
                X_country_exog = pd.DataFrame()

            if not data_wide_global_exog.empty:
                X_global_exog = pd.concat([
                    data_wide_global_exog[regressor_col].shift(lag-1)
                    for lag in self.global_exog_lags
                    for regressor_col in data_wide_global_exog.columns
                    ],
                    axis=1,
                    keys=[
                        f"{regressor_col}_lag_{lag}"
                        for lag in self.global_exog_lags
                        for regressor_col in data_wide_global_exog.columns
                    ]
                )
            else:
                X_global_exog = pd.DataFrame()

            to_concatenate = [X, X_country_exog, X_global_exog]
            to_concatenate = [df for df in to_concatenate if not df.empty]
            all_X = pd.concat(to_concatenate, axis=1)
            all_X = all_X.iloc[max(self.lags):]

            # X = pd.concat(
            #     [
            #         # Shift by 1 fewer lag because we are predicting the next period
            #         # This is also the reason why we add 3 months to the index later
            #         data_wide[regressor_col].shift(lag - 1) 
            #         for lag in self.lags
            #         for regressor_col in data_wide.columns
            #     ],
            #     axis = 1,
            #     keys = [
            #         f"{regressor_col}_lag_{lag}"
            #         for lag in self.lags
            #         for regressor_col in data_wide.columns
            #     ]
            # )
            # X = X.iloc[max(self.lags):]

            predictions[col] = self.models[col].predict(sm.add_constant(all_X))

        predictions = pd.DataFrame(predictions)
        
        # Convert the predictions back to long format
        if self.long_format:
            predictions = predictions.stack().reset_index()
            predictions.columns = [self.date_column, "Country", "inflation"]
            # The predictions are one quarter ahead
            predictions[self.date_column] += pd.DateOffset(months=3)
        else:
            predictions.index += pd.DateOffset(months=3)

        return predictions
