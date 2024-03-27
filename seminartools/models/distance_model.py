import pandas as pd
import numpy as np
import bambi as bmb
from .base_model import BaseModel
import arviz as az


class DistanceModel(BaseModel):
    def __init__(
        self,
        lags: int = 1,
        country_column: str = "country",
        target_column: str = "inflation",
        exogenous_columns: list = [
            "gdp_growth",
            "interest_rate",
            "unemployment_rate",
            "commodity_CRUDE_PETRO",
            "commodity_iNATGAS",
            "commodity_iAGRICULTURE",
            "commodity_iMETMIN",
            "commodity_iPRECIOUSMET",
        ],
        tune: int = 500,
    ):
        """
        Initializes the model.
        """
        self.lags = lags
        self.country_column = country_column
        self.target_column = target_column
        self.exogenous_columns = exogenous_columns

        # Regress on lagged exogenous variables and lagged target variable
        self.lagged_exog_columns = [
            f"{col}_lag_{i}"
            for i in range(1, lags + 1)
            for col in exogenous_columns + [target_column]
        ]

        # MCMC parameters
        self.tune = tune

    def fit(self, data: pd.DataFrame):
        """
        Fits the model on data.
        """
        # Get country out of the multiindex
        data = data.reset_index(level=self.country_column)

        countries = data[self.country_column].unique()
        if len(countries) <= 1:
            print(
                "WARNING: Only one country in the data in RandomEffectsModel.fit(). Random effects model not applicable."
            )

        assert (
            self.target_column in data.columns
        ), f"Target column {self.target_column} not in data."
        assert all(
            [col in data.columns for col in self.exogenous_columns]
        ), f"Exogenous columns not in data."

        # Create lagged variables
        data = self._create_lagged_variables(data)

        # Drop rows with NaNs due to lagged variables
        data = data.dropna()



    def _create_lagged_variables(self, data: pd.DataFrame):
        """
        Creates lagged variables for the model.

        Takes into account that the data is grouped by country.
        """

        def _add_lags(country_df):
            for i in range(1, self.lags + 1):
                for col in self.exogenous_columns + [self.target_column]:
                    country_df[f"{col}_lag_{i}"] = country_df[col].shift(
                        i, freq=pd.DateOffset(months=3)
                    )

            return country_df

        return data.groupby(self.country_column, group_keys=False).apply(_add_lags)

    def predict(self, data: pd.DataFrame, pointwise_aggregation_method: str = "mean"):
        """
        Predicts the target variable on data.
        """
        # Get country out of the multiindex
        data = data.reset_index(level=self.country_column)

        data = self._create_lagged_variables(data)
        data = data.dropna()

        # # Predict using the model that was fit
        # predictions = self.model.predict(self.results, data=data, inplace=False)

        # if pointwise_aggregation_method == "mean":
        #     predictions = az.extract(predictions)[f"{self.target_column}_mean"].mean(
        #         "sample"
        #     )
        # else:
        #     raise ValueError(
        #         f"Unknown pointwise aggregation method: {pointwise_aggregation_method}"
        #     )

        # data["predictions"] = predictions

        # predictions = (
        #     data.reset_index()
        #     .set_index(["date", self.country_column])["predictions"]
        #     .rename("inflation")
        #     .to_frame()
        # )
        # # only keep the predictions for the last date
        # end_date = predictions.index.get_level_values("date").max()
        # predictions = predictions.loc[end_date:end_date]

        # # add 3 months to prediction.get_level_values(0)
        # predictions.index = pd.MultiIndex.from_tuples(
        #     zip(
        #         predictions.index.get_level_values(0) + pd.DateOffset(months=3),
        #         predictions.index.get_level_values(1),
        #     )
        # )

        return predictions
