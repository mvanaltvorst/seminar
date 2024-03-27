import pandas as pd
import numpy as np
from .base_model import BaseModel
from seminartools.utils import geo_distance
import arviz as az
import pymc as pm
from seminartools.matern_kernel import MaternGeospatial
import pytensor



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
        draws: int = 1500,
        chains: int = 4,
        distance_function: callable = geo_distance,
        distance_scaling: float = 1000,
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
        self.draws = draws
        self.chains = chains

        # A scaling such that the matern covariance function is not too small
        self.distance_scaling = distance_scaling

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

        # Create lagged variables
        data = self._create_lagged_variables(data)

        # Drop rows with NaNs due to lagged variables
        data = data.dropna()

        with pm.Model(
            coords={
                "country": self.countries,
                "countrytime": data.index,
                "exogenous": self.exogenous_columns,
                "target": [self.target_column],
            }
        ) as self.model:
            country_mean_tau = pm.Gamma("country_mean_tau", alpha=1, beta=1)
            country_means = pm.Normal(
                "country_means", mu=0, tau=country_mean_tau, dims="country"
            )

            regression_coefficient_tau = pm.Gamma(
                "regression_coefficient_tau", alpha=1, beta=1
            )
            regression_coefficients = pm.Normal(
                "regression_coefficients",
                mu=0,
                tau=regression_coefficient_tau,
                dims=["exogenous", "country"],
            )
            # Index of the country in the countries array
            # Allows us to select the correct column
            # from the regression coefficients.
            country_index_vector = (
                np.array([np.where(self.countries == country)[0][0] for country in data[self.country_column]])
            )

            # New GP part
            ls = pm.Gamma("ls", alpha=2, beta=10000)  # Example length scale prior
            cov_func = MaternGeospatial(
                2,
                ls,
                distance_scaling=self.distance_scaling,
                distance_function=self.distance_function,
            )

            latent = pm.gp.Latent(cov_func=cov_func)

            country_correlated_noise = latent.prior(
                "country_correlated_noise", dims="country", X=self.countries
            )

            # Incorporate the GP into your model's likelihood
            sigma = pm.HalfCauchy("sigma", beta=1)
            # Likelihood
            # mu = country_means + pm.math.dot(
            #     data[self.exogenous_columns].values, regression_coefficients
            # ) + country_correlated_noise
            # mu = pm.math.dot(
            #     data[self.exogenous_columns].values, regression_coefficients[:, country_index_vector]
            # )


            mu = (data[self.exogenous_columns].values * regression_coefficients[:, country_index_vector].T).sum(axis=1)

            y_obs = pm.Normal(
                "likelihood",
                mu=mu,
                sigma=sigma,
                observed=data[self.target_column].values,
                dims="countrytime",
            )

            self.trace = pm.sample(tune=self.tune, draws=self.draws, chains=self.chains, cores=self.chains)


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

        # return predictions
