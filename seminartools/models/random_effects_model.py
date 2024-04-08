import pandas as pd
import numpy as np
import bambi as bmb
from .base_model import BaseModel
import arviz as az
from xarray import apply_ufunc
from scipy.stats import gaussian_kde
from scipy.integrate import simps


class RandomEffectsModel(BaseModel):
    def __init__(
        self,
        lags: int = 1,
        date_column: str = "date",
        country_column: str = "country",
        target_column: str = "inflation",
        exogenous_columns: list = [],
        tune: int = 500,
        chains: int = 4,
        num_draws: int = 1500,
        nuts_sampler: str = "nutpie",  # ❤️ nutpie
        pointwise_aggregation_method: str = "mean",
    ):
        """
        Initializes the model.
        """
        self.lags = lags
        self.date_column = date_column
        self.country_column = country_column
        self.target_column = target_column
        self.exogenous_columns = exogenous_columns
        self.pointwise_aggregation_method = pointwise_aggregation_method

        # MCMC parameters
        self.tune = tune
        self.chains = chains
        self.num_draws = num_draws
        self.nuts_sampler = nuts_sampler

        # Build the formula to be used by Bambi
        self.formula = f"{target_column} ~ (1 | {country_column}) + "

        # Regress on lagged exogenous variables and lagged target variable
        lagged_exog_columns = [
            f"{col}_lag_{i}"
            for i in range(1, lags + 1)
            for col in exogenous_columns + [target_column]
        ]

        self.regression_columns = lagged_exog_columns

        # self.formula += " + ".join(lagged_exog_columns)
        for i, col in enumerate(lagged_exog_columns):
            self.formula += f"(0 + {col} | {country_column})"
            if i < len(lagged_exog_columns) - 1:
                self.formula += " + "

    def fit(self, data: pd.DataFrame):
        """
        Fits the model on data.
        """

        data = data.set_index(self.date_column)

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

        # Normalize
        self.feature_means = data[self.regression_columns].mean()
        self.feature_stds = data[self.regression_columns].std()
        self.target_mean = data[self.target_column].mean()
        self.target_std = data[self.target_column].std()

        data[self.regression_columns] = (data[self.regression_columns] - self.feature_means) / self.feature_stds
        data[self.target_column] = (data[self.target_column] - self.target_mean) / self.target_std

        # Fit the model
        self.model = bmb.Model(self.formula, data=data)
        self.results = self.model.fit(
            draws=self.num_draws,
            tune=self.tune,
            chains=self.chains,
            cores=self.chains,
            nuts_sampler=self.nuts_sampler,
        )

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

    def predict(self, data: pd.DataFrame):
        """
        Predicts the target variable on data.
        """
        data = data.set_index(self.date_column)

        data = self._create_lagged_variables(data)
        data = data.dropna()

        #normalize
        data[self.regression_columns] = (data[self.regression_columns] - self.feature_means) / self.feature_stds

        # Predict using the model that was fit
        predictions = self.model.predict(
            self.results, data=data, inplace=False, sample_new_groups=True
        )

        if self.pointwise_aggregation_method == "mean":
            predictions = az.extract(predictions)[f"{self.target_column}_mean"].mean(
                "sample"
            )
            
        elif self.pointwise_aggregation_method == "distribution":
            def getDistribution(row):
                kde = gaussian_kde(row)
                return kde

            predictions = az.extract(predictions)[f"{self.target_column}_mean"]
            df = predictions.to_dataframe()
            df = df.groupby("inflation_obs")
            df = df.agg(getDistribution)
            predictions = df["inflation_mean"].values

        else:
            raise ValueError(
                f"Unknown pointwise aggregation method: {self.pointwise_aggregation_method}"
            )
        
        data["predictions"] = predictions
        predictions = (
            data.reset_index()
            .set_index([self.date_column, self.country_column])["predictions"]
            .rename("inflation")
            .to_frame()
        )
        # only keep the predictions for the last date
        end_date = predictions.index.get_level_values(self.date_column).max()
        predictions = predictions.loc[end_date:end_date]

        predictions.index = pd.MultiIndex.from_tuples(
            zip(
                predictions.index.get_level_values(0) + pd.DateOffset(months=3),
                predictions.index.get_level_values(1),
            )
        )
        
        # deNormalize
        if self.pointwise_aggregation_method == "mean":

            predictions.inflation = predictions.inflation * self.target_std + self.target_mean
            
        if self.pointwise_aggregation_method == "distribution":

            def denormalize_density(kde):
                x_axis = np.linspace(min(kde.dataset[0]),max(kde.dataset[0]), 1000)
                pdf_values = kde.pdf(x_axis)
                denormalized_x_axis = x_axis * self.target_std + self.target_mean
                area = simps(pdf_values)
                pdf_values = pdf_values/area
                return {"pdf": pdf_values, "inflation_grid": denormalized_x_axis}

            predictions.inflation = predictions.inflation.apply(denormalize_density)

        return predictions.reset_index().rename(
            columns={"level_0": self.date_column, "level_1": self.country_column}
        )
