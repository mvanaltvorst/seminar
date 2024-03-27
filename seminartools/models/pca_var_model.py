from seminartools.models.base_model import BaseModel
from .var_model import VARModel
import pandas as pd
import numpy as np


class PCAVARModel(BaseModel):
    """
    Projects inflation rates onto a lower dimensional space using PCA, then uses a VAR model to forecast these principal components and convert them back to the original space.


    Args:
        country_column (str): The name of the column containing the country names. Used to split the data into countries.
        inflation_column (str): The name of the column containing the inflation rates.
        num_pcs (int): The number of principal components to use.
        lags (list[int]): The lags to use in the PCA VAR model.
        standardize_pre_post (bool): 
            if True, we standardize the data before fitting the model and revert the standardization after predicting.
            This way, the covariance we perform PCA on is basically a correlation matrix.
            This should help with avoiding single countries dominating the PCA.
    """

    def __init__(
        self,
        country_column: str = "Country",
        date_column: str = "yearmonth",
        inflation_column: str = "inflation",
        num_pcs: int = 3,
        country_exogenous_columns: list[str] = [],
        global_exogenous_columns: list[str] = [],
        lags: list[int] = [1, 2, 3, 4],
        country_exog_lags: list[int] = [1],
        global_exog_lags: list[int] = [1],
        standardize_pre_post: bool = False
    ):
        self.country_column = country_column
        self.date_column = date_column
        self.inflation_column = inflation_column
        self.num_pcs = num_pcs
        self.country_exogenous_columns = country_exogenous_columns
        self.global_exogenous_columns = global_exogenous_columns
        self.lags = lags
        self.country_exog_lags = country_exog_lags
        self.global_exog_lags = global_exog_lags
        self.standardize_pre_post = standardize_pre_post


    def fit(self, data: pd.DataFrame):
        """
        Fit the model to the data.
        """
        # Step 0: convert into wide format. Index is yearmonth, column is Country, values are `inflation`.
        data_wide = data.pivot(index=self.date_column, columns=self.country_column, values=self.inflation_column)

        # Need to store the columns of the wide data for during prediction
        self.data_wide_columns = data_wide.columns

        # Step 0.5: standardize the data
        if self.standardize_pre_post:
            self.means = data_wide.mean()
            self.stds = data_wide.std()
            data_wide = (data_wide - self.means) / self.stds

        # Step 1: factor decomposition
        self.all_eigenvalues, self.all_eigenvectors = np.linalg.eig(data_wide.cov())

        # We only consider the first num_pcs principal components
        # and the corresponding eigenvalues
        self.eigenvalues = self.all_eigenvalues[: self.num_pcs]
        self.eigenvectors = self.all_eigenvectors[:, : self.num_pcs]

        # We project the data onto the first num_pcs principal components
        pcs = data_wide.to_numpy() @ self.eigenvectors
        pcs = pd.DataFrame(
            pcs,
            index=data_wide.index,
            columns=[f"PC{i}" for i in range(1, self.num_pcs + 1)],
        )
        pc_columns = pcs.columns.values.tolist()

        # Add exogenous variables to the principal components
        # Useable for VAR-X model
        # Put all data in wide format
        data_wide_country_exog = data.pivot(index=self.date_column, columns=self.country_column, values=self.country_exogenous_columns)
        data_wide_country_exog.columns = data_wide_country_exog.columns.map('_'.join)
        country_exog_columns = data_wide_country_exog.columns.values.tolist()
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
        global_exog_columns = data_wide_global_exog.columns.values.tolist()

        to_concatenate = [pcs, data_wide_country_exog, data_wide_global_exog]
        to_concatenate = [df for df in to_concatenate if not df.empty]
        all_X = pd.concat(to_concatenate, axis=1)
        all_X = all_X.iloc[max(self.lags):]

        # Step 2: fit a VAR model to the principal components
        # long_format = False because PCS is already in wide format
        self.var = VARModel(
            inflation_column=pc_columns,
            country_exogenous_columns=country_exog_columns,
            global_exogenous_columns=global_exog_columns,
            date_column=self.date_column,
            lags=self.lags,
            country_exog_lags=self.country_exog_lags,
            global_exog_lags=self.global_exog_lags,
            long_format=False
        )
        self.var.fit(all_X)

        # # Step 2: fit a VAR model to the principal components
        # # long_format = False because PCS is already in wide format
        # self.var = VARModel(self.lags, long_format=False)
        # self.var.fit(pcs)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict the inflation rate for the next month.
        """
        data_wide = data.pivot(index=self.date_column, columns=self.country_column, values=self.inflation_column)
        data_wide = data_wide[self.data_wide_columns]

        # Step 0.5: standardize the data
        if self.standardize_pre_post:
            data_wide = (data_wide - self.means) / self.stds

        # Step 1: project the data onto the first num_pcs principal components
        pcs = data_wide.to_numpy() @ self.eigenvectors
        pcs = pd.DataFrame(
            pcs,
            index=data_wide.index,
            columns=[f"PC{i}" for i in range(1, self.num_pcs + 1)],
        )

        # Add exogenous variables to the principal components
        # Useable for VAR-X model
        # Put all data in wide format
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

        to_concatenate = [pcs, data_wide_country_exog, data_wide_global_exog]
        to_concatenate = [df for df in to_concatenate if not df.empty]
        all_X = pd.concat(to_concatenate, axis=1)
        all_X = all_X.iloc[max(self.lags):]

        # Step 2: forecast the principal components
        pc_predictions = self.var.predict(all_X)

        # Step 3: convert the principal components back to the original space
        predictions = pc_predictions.to_numpy() @ self.eigenvectors.T
        predictions = pd.DataFrame(
            predictions,
            index=pc_predictions.index,
            columns=data_wide.columns,
        )

        # Step 3.5: revert the standardization
        if self.standardize_pre_post:
            predictions = predictions * self.stds + self.means

        # Step 4: convert the predictions back to long format
        predictions = predictions.stack().reset_index().rename(columns={0: "inflation"})

        # Only retain final month
        predictions = predictions[predictions[self.date_column] == predictions[self.date_column].max()]

        return predictions

