import pandas as pd
import numpy as np
from .base_model import BaseModel
from seminartools.utils import geo_distance
import arviz as az
import pymc as pm
from seminartools.matern_kernel import MaternGeospatial
from tqdm import tqdm
from scipy.stats import gaussian_kde
from scipy.integrate import simps



class DistanceModel(BaseModel):
    def __init__(
        self,
        lags: int = 1,
        date_column: str = "date",
        country_column: str = "country",
        target_column: str = "inflation",
        exogenous_columns: list = [],
        tune: int = 500,
        draws: int = 1500,
        chains: int = 4,
        distance_function: callable = geo_distance,
        distance_scaling: float = 1000,
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
        self.distance_function = distance_function
        self.pointwise_aggregation_method = pointwise_aggregation_method

        # Regress on lagged exogenous variables and lagged target variable
        self.lagged_exog_columns = [
            f"{col}_lag_{i}"
            for i in range(1, lags + 1)
            for col in exogenous_columns + [target_column]
        ]

        self.regression_columns = self.lagged_exog_columns

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
        data = data.set_index(self.date_column)

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

        # Normalize
        self.feature_means = data[self.regression_columns].mean()
        self.feature_stds = data[self.regression_columns].std()
        self.target_mean = data[self.target_column].mean()
        self.target_std = data[self.target_column].std()

        data[self.regression_columns] = (
            data[self.regression_columns] - self.feature_means
        ) / self.feature_stds
        data[self.target_column] = (data[self.target_column] - self.target_mean) / self.target_std

        self.times = data.index.unique().sort_values()

        with pm.Model(
            coords={
                "country": self.countries,
                "time": self.times,
                "regression_columns": self.regression_columns,
                "target": [self.target_column],
                "countrytime": data.index,
            }
        ) as self.model:
            # Index of the country in the countries array
            # Allows us to select the correct column
            # from the regression coefficients.
            country_index_vector = np.array(
                [
                    np.where(self.countries == country)[0][0]
                    for country in data[self.country_column]
                ]
            )

            # New GP part
            ls = pm.Gamma("ls", alpha=20, beta=0.1)  # Example length scale prior
            cov_func = MaternGeospatial(
                2,
                ls,
                distance_scaling=self.distance_scaling,
                distance_function=self.distance_function,
            )

            latent = pm.gp.Latent(cov_func=cov_func)

            # dim: num_countries
            country_means = latent.prior("intercepts", X=self.countries)

            regression_coefficients = []
            for regression_col in tqdm(
                self.regression_columns, desc="Creating regression coefficients"
            ):
                regression_coefficients.append(
                    latent.prior(f"regression_coefficients_{regression_col}", X=self.countries)
                )

            # dim: num_regression_coefficients x num_countries
            regression_coefficients = pm.math.stack(regression_coefficients)

            # Incorporate the GP into the model's likelihood
            sigma = pm.HalfCauchy("sigma", beta=1)

            # We calculate y_t = X_t' beta_{country(t)}
            # in a vectorized way by doing elementwise
            # multiplication of the regression variables
            # with the indexed regression coefficients
            # and summing over the columns.
            mu = (
                data[self.regression_columns].values
                * regression_coefficients[:, country_index_vector].T
            ).sum(axis=1)
            # Then we add the constant per country
            mu += country_means[country_index_vector]

            y_obs = pm.Normal(
                "likelihood",
                mu=mu,
                sigma=sigma,
                observed=data[self.target_column].values,
                dims="countrytime",
            )

            self.trace = pm.sample(
                tune=self.tune,
                draws=self.draws,
                chains=self.chains,
                cores=self.chains,
                nuts_sampler="nutpie",
            )

        # We stack the chains s.t. we obtain a shape of [(chains * samples), countries]
        self.country_intercepts = np.concatenate(
            [
                self.trace.posterior["intercepts"].values[i]
                for i in range(self.trace.posterior["intercepts"].values.shape[0])
            ],
            axis=0,
        )

        # [(chains * samples), countries, num_regression]
        self.regression_coefficients = np.stack(
            [
                np.concatenate(
                    [
                        self.trace.posterior[f"regression_coefficients_{regression_col}"].values[
                            i
                        ]
                        for i in range(
                            self.trace.posterior[
                                f"regression_coefficients_{regression_col}"
                            ].values.shape[0]
                        )
                    ],
                    axis=0,
                )
                for regression_col in self.regression_columns
            ],
            axis=-1,
        )

        # Append the mean intercept and coefficients as new columns
        # in case we want to predict missing countries in the .predict() method
        mean_intercepts = np.mean(self.country_intercepts, axis=1, keepdims=True)
        mean_coefficients = np.mean(self.regression_coefficients, axis=1, keepdims=True)

        self.country_intercepts = np.concatenate([self.country_intercepts, mean_intercepts], axis=1)
        self.regression_coefficients = np.concatenate([self.regression_coefficients, mean_coefficients], axis=1)


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

    def predict(self, data: pd.DataFrame, debug: bool = False):
        """
        Predicts the target variable on data, vectorized for improved performance.
        """
        # Prepare data
        data = data.set_index(self.date_column)
        data = self._create_lagged_variables(data)
        data = data.dropna()
        data = data[
            data.index == data.index.max()
        ]  # Only use the latest time period for predictions

        # normalize
        data[self.regression_columns] = (
            data[self.regression_columns] - self.feature_means
        ) / self.feature_stds

        # Ensure correct order of countries to match model's country order
        data["country_idx"] = data[self.country_column].map(
            {country: idx for idx, country in enumerate(self.countries)}
        ).fillna(len(self.countries))  # `len(self.countries)` points to the new "mean country" coefficients
        data["country_idx"] = data["country_idx"].astype(int)

        # Prepare regression features from data
        regression_data = data[self.regression_columns].values  # Shape: [rows, regression_cols]

        # Compute predictions
        # Multiply exog_data (broadcasted to [samples, rows, exogenous]) with regression_coefficients
        # Sum over the exogenous dimension, then add intercepts
        country_indices = data["country_idx"].values
        predictions = (
            regression_data[None, :, :] * self.regression_coefficients[:, country_indices, :]
        ).sum(axis=-1) + self.country_intercepts[:, country_indices]
        # predictions shape: [samples, rows]

        # Aggregate predictions across samples
        if self.pointwise_aggregation_method == "mean":
            aggregated_predictions = predictions.mean(axis=0)
        elif self.pointwise_aggregation_method == "median":
            aggregated_predictions = np.median(predictions, axis=0)
        elif self.pointwise_aggregation_method == "distribution":
            def getKDE(row):
                kde = gaussian_kde(row)
                return kde

            aggregated_predictions = np.apply_along_axis(getKDE, arr=predictions, axis=0)

        else:
            raise ValueError(f"Unsupported aggregation method: {self.pointwise_aggregation_method}")

        predictions_df = pd.DataFrame(
            {
                self.country_column: data[self.country_column].values,
                "inflation": aggregated_predictions,
            }
        )

        # for debugging purposes, we concat with the used regression coefficients
        if debug:
            predictions_df = pd.concat(
                [
                    predictions_df,
                    pd.DataFrame(
                        self.regression_coefficients[:, country_indices, :].mean(axis=0),
                        columns=[f"{col}_coefficient" for col in self.regression_columns],
                    ),
                ],
                axis=1,
            )

        predictions_df[self.date_column] = data.index.max() + pd.DateOffset(months=3)
        predictions_df = predictions_df

        # Denormalize
        if self.pointwise_aggregation_method == "mean" or self.pointwise_aggregation_method == "median":
            predictions_df["inflation"] = predictions_df["inflation"] * self.target_std + self.target_mean

        elif self.pointwise_aggregation_method == "distribution":
            def denormalize_density(kde):
                x_axis = np.linspace(min(kde.dataset[0]),max(kde.dataset[0]), 1999)
                pdf_values = kde.pdf(x_axis)
                denormalized_x_axis = x_axis * self.target_std + self.target_mean
                area = simps(pdf_values)
                pdf_values = (pdf_values/area)
                return {"pdf": pdf_values, "inflation_grid": denormalized_x_axis}
            
            predictions_df.inflation = predictions_df.inflation.apply(denormalize_density)
        return predictions_df
