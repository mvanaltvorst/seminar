import pandas as pd
import numpy as np
from .base_model import BaseModel
from seminartools.utils import geo_distance
import arviz as az
import pymc as pm
from pymc.gp.cov import Matern32


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
        distance_function: callable = geo_distance,
    ):
        """
        Initializes the model.
        """
        self.lags = lags
        self.country_column = country_column
        self.target_column = target_column
        self.exogenous_columns = exogenous_columns
        self.distance_function = distance_function

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

        self.countries = data[self.country_column].unique()
        if len(self.countries) <= 1:
            print(
                "WARNING: Only one country in the data in RandomEffectsModel.fit(). Random effects model not applicable."
            )

        assert (
            self.target_column in data.columns
        ), f"Target column {self.target_column} not in data."
        assert all(
            [col in data.columns for col in self.exogenous_columns]
        ), f"Exogenous columns not in data."

        self.distance_matrix = self._geographical_distance_matrix(self.countries)

        # Create lagged variables
        data = self._create_lagged_variables(data)

        # Drop rows with NaNs due to lagged variables
        data = data.dropna()

        with pm.Model(
            coords={
                "country": self.countries,
                "exogenous": self.exogenous_columns,
                "target": [self.target_column],
            }
        ) as self.model:
            constant_mean_tau = pm.Gamma("constant_mean_tau", alpha=1, beta=1)
            constant_mean = pm.Normal("constant_mean", mu=0, tau=constant_mean_tau)

            country_mean_tau = pm.Gamma("country_mean_tau", alpha=1, beta=1)
            country_means = pm.Normal(
                "country_means", mu=constant_mean, tau=country_mean_tau, dims="country"
            )

            regression_coefficient_mean_tau = pm.Gamma(
                "regression_coefficient_mean_tau", alpha=1, beta=1
            )
            regression_coefficient_mean = pm.Normal(
                "regression_coefficient_mean",
                mu=0,
                tau=regression_coefficient_mean_tau,
                dims="exogenous",
            )

            regression_coefficient_tau = pm.Gamma(
                "regression_coefficient_tau", alpha=1, beta=1
            )
            regression_coefficients = pm.Normal(
                "regression_coefficients",
                mu=regression_coefficient_mean,
                tau=regression_coefficient_tau,
                dims=["exogenous", "target"],
            )

            # Likelihood
            mu = country_means + pm.math.dot(
                data[self.exogenous_columns].values, regression_coefficients
            )
            sigma = pm.HalfCauchy("sigma", beta=1)

            # New GP part
            ls = pm.Gamma("ls", alpha=2, beta=1)  # Example length scale prior
            # cov_func = Matern32Chordal(ls=ls)
            cov_func = Matern32(2, ls)
            cov_matrix = cov_func.full_from_distance(self.distance_matrix)

            latent = pm.gp.Latent(cov_func=cov_matrix)

            country_correlated_noise = latent.prior(
                "country_correlated_noise", dims="country", X=self.countries
            )

            # Incorporate the GP into your model's likelihood
            sigma = pm.HalfCauchy("sigma", beta=1)
            y_obs = pm.Normal(
                "likelihood",
                mu=mu + country_correlated_noise,
                sigma=sigma,
                observed=data[self.target_column].values,
                dims="country",
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

    def _geographical_distance_matrix(self, countries):
        """
        Create a matrix of geographical distances between countries based on their coordinates.
        """
        distance_matrix = np.zeros((len(countries), len(countries)))

        for i, country_i in enumerate(countries):
            for j, country_j in enumerate(countries):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    distance_matrix[i, j] = self.distance_function(country_i, country_j)
        return distance_matrix

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

        # return predictions
