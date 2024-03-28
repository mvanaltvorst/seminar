import pandas as pd
import numpy as np
from .base_model import BaseModel
from seminartools.utils import geo_distance
import arviz as az
import pymc as pm
from seminartools.matern_kernel import MaternGeospatial
from tqdm import tqdm


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

        self.times = data.index.unique().sort_values()

        with pm.Model(
            coords={
                "country": self.countries,
                "time": self.times,
                "exogenous": self.exogenous_columns,
                "target": [self.target_column],
                "countrytime": data.index,
            }
        ) as self.model:
            # country_mean_tau = pm.Gamma("country_mean_tau", alpha=1, beta=1)
            # country_means = pm.Normal(
            #     "country_means", mu=0, tau=country_mean_tau, dims="country"
            # )

            # regression_coefficient_tau = pm.Gamma(
            #     "regression_coefficient_tau", alpha=1, beta=1
            # )
            # regression_coefficients = pm.Normal(
            #     "regression_coefficients",
            #     mu=0,
            #     tau=regression_coefficient_tau,
            #     dims=["exogenous", "country"],
            # )

            # Index of the country in the countries array
            # Allows us to select the correct column
            # from the regression coefficients.
            country_index_vector = np.array(
                [
                    np.where(self.countries == country)[0][0]
                    for country in data[self.country_column]
                ]
            )
            time_index_vector = np.array(
                [np.where(self.times == time)[0][0] for time in data.index]
            )

            total_coordinates = np.array(
                [(time, country) for time in self.times for country in self.countries]
            )

            # New GP part
            ls = pm.Gamma("ls", alpha=2, beta=1)  # Example length scale prior
            cov_func = MaternGeospatial(
                2,
                ls,
                distance_scaling=self.distance_scaling,
                distance_function=self.distance_function,
            )

            # country_time_noises = []
            # for time in tqdm(self.times):
            #     gp_noise = latent.prior(f"gp_noise_{time}", X=self.countries)
            #     # Indexing to assign the generated noise to the correct positions
            #     country_time_noises.append(gp_noise)

            # country_time_noise = pytensor.stack(country_time_noises)
            # print(f"{country_time_noise.eval().shape=}")

            latent = pm.gp.Latent(cov_func=cov_func)

            # dim: num_countries
            country_means = latent.prior("intercepts", X=self.countries)

            regression_coefficients = []
            for exog in tqdm(
                self.exogenous_columns, desc="Creating regression coefficients"
            ):
                regression_coefficients.append(
                    latent.prior(f"regression_coefficients_{exog}", X=self.countries)
                )

            # dim: num_regression_coefficients x num_countries
            regression_coefficients = pm.math.stack(regression_coefficients)

            # Incorporate the GP into the model's likelihood
            sigma = pm.HalfCauchy("sigma", beta=1)

            # We calculate y_t = X_t' beta_{country(t)}
            # in a vectorized way by doing elementwise
            # multiplication of the exogenous variables
            # with the indexed regression coefficients
            # and summing over the columns.
            mu = (
                data[self.exogenous_columns].values
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
                tune=self.tune, draws=self.draws, chains=self.chains, cores=self.chains
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

    def predict(self, data: pd.DataFrame, aggregation_method: str = "mean"):
        """
        Predicts the target variable on data.
        """
        # Get country out of the multiindex
        data = data.reset_index(level=self.country_column)

        data = self._create_lagged_variables(data)
        data = data.dropna()

        # Map countries to their indices in the model
        country_to_index = {country: idx for idx, country in enumerate(self.countries)}

        # Prepare a container for the predictions from all samples
        all_predictions = []

        # Number of samples in the trace
        num_samples = len(self.trace.posterior["intercepts"].values[0])

        all_country_intercepts = self.trace.posterior["intercepts"].values[0]
        all_regression_coefficients = {
            exog: self.trace.posterior[f"regression_coefficients_{exog}"].values[0]
            for exog in self.exogenous_columns
        }

        # Iterate over each sample
        for sample in range(num_samples):
            # Extract the intercepts and coefficients for this sample
            country_intercepts = all_country_intercepts[sample, :]
            regression_coefficients = {
                exog: all_regression_coefficients[exog][sample]
                for exog in self.exogenous_columns
            }

            # Calculate predictions for all final rows in data
            for _, row in data[data.index == data.index.max()].iterrows():
                country_idx = country_to_index[row[self.country_column]]
                intercept = country_intercepts[country_idx]

                # Calculate the contribution from each exogenous variable for this sample
                exog_contribution = sum(
                    row[exog] * regression_coefficients[exog][country_idx]
                    for exog in self.exogenous_columns
                )

                # The prediction for this row and sample
                prediction = intercept + exog_contribution
                all_predictions.append(
                    {"country": row[self.country_column], "prediction": prediction}
                )

        # Convert the list of lists into a 2D numpy array for easier manipulation
        all_predictions = pd.DataFrame(all_predictions)

        # Aggregate predictions across samples
        if aggregation_method == "mean":
            aggregated_predictions = all_predictions.groupby("country").mean()
        elif aggregation_method == "median":
            aggregated_predictions = all_predictions.groupby("country").median()
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

        # Create a DataFrame for the aggregated predictions
        predictions_df = aggregated_predictions.rename(
            columns={"prediction": "inflation"}
        ).reset_index()
        predictions_df["date"] = data.index.max() + pd.DateOffset(months=3)
        predictions_df = predictions_df.set_index(["date", "country"])

        return predictions_df
