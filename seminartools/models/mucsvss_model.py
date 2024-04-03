#import numpy as np
from .base_model import BaseModel
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.integrate import simps
from scipy.signal import fftconvolve
from ..utils import geo_distance
import jax.numpy as jnp
from jax import random, jit, vmap, grad
from jax.scipy.stats import jnorm
from jax.scipy.linalg import cholesky

# num cpu
from multiprocessing import cpu_count

# nu follows a normal distribution with variance gamma
GAMMA = 0.2

# zeta follows a normal distribution with variance theta
THETA = 0.002

INIT_LNSETASQ = 0
INIT_LNSEPSILONSQ = 0
INIT_DELTA = 0
INIT_TAU = 0

VAGUE_PRIOR_LNSETASQ_SIGMA = 3
VAGUE_PRIOR_LNSEPSILONSQ_SIGMA = 3
VAGUE_PRIOR_DELTA_SIGMA = 1
VAGUE_PRIOR_TAU_SIGMA = 1

N_CORES = cpu_count()


# Multivariate Unobserved Component Stochastic Volatility Stochastic Seasonality Particle Filter
class MUCSVSSModel(BaseModel):
    """
    Same as UCSVSS model, but we let (epsilon, eta, xi) follow
    multivariate normal distribution where the covariance matrix is /not/ diagonal
    anymore.

    sigma_{i,j,t} = rho_{i,j} * sigma_{i,t} * sigma_{j,t}

    where rho_{i,j} is constant over time and a function of distance between i and j.
    via e.g. the Matern kernel.
    """

    # We benefit from fitting on the entire dataset before predicting
    # Uses intermediate states to quickly evaluate the model on new data
    REQUIRES_ANTE_FULL_FIT = True

    def __init__(
        self,
        num_particles: int,
        stochastic_seasonality: bool,
        country_column: str = "country",
        date_column: str = "date",
        inflation_column: str = "inflation",
        distance_function: callable = lambda x, y: geo_distance(x, y) / 500,
    ):
        self.distance_function = distance_function
        self.num_particles = num_particles
        self.country_column = country_column
        self.date_column = date_column
        self.inflation_column = inflation_column

        self.gamma = GAMMA

        self.init_lnsetasq = INIT_LNSETASQ
        self.init_lnsepsilonsq = INIT_LNSEPSILONSQ
        self.init_delta = INIT_DELTA
        self.init_tau = INIT_TAU

        self.vague_prior_lnsetasq_sigma = VAGUE_PRIOR_LNSETASQ_SIGMA
        self.vague_prior_lnsepsilonsq_sigma = VAGUE_PRIOR_LNSEPSILONSQ_SIGMA

        # No stochastic seasonality is equivalent to theta = 0 and a prior with 0 variance
        if stochastic_seasonality:
            self.vague_prior_delta_sigma = VAGUE_PRIOR_DELTA_SIGMA
            self.theta = THETA
        else:
            self.vague_prior_delta_sigma = 0
            self.theta = 0

        self.vague_prior_tau_sigma = VAGUE_PRIOR_TAU_SIGMA

    def fit(self, data: pd.DataFrame):
        """
        This model is not meant to be fitted to data every iteration using this method.
        Preferably, fit once on the entire dataset using `run_pf` and then
        figure out the historical predictions using the historical particles.
        """
        if not hasattr(self, "stored_state_means"):
            raise ValueError("Model has not been `run_pf`'d yet.")

    def full_fit(self, data: pd.DataFrame, aggregation_method: str = "median"):
        """
        Run the particle filter on the data of a single country.
        """
        self.aggregation_method = aggregation_method

        # dfs = Parallel(n_jobs=N_CORES)(
        #     delayed(self._run_pf)(data.loc[data[self.country_column] == country])
        #     for country in tqdm(data[self.country_column].unique())
        # )
        df = self._run_pf(data)
        # self.stored_state_means = pd.concat(dfs, axis=0)
        self.stored_state_means = df



    def _run_pf(self, data: pd.DataFrame):
        """
        Run the particle filter on the data of a single country.
        """
        key = random.PRNGKey(42)
        key, subkey = random.split(key)

        self.countries = data[self.country_column].unique().tolist()  # n countries
        n = len(self.countries)
        self.times = sorted(data[self.date_column].unique().tolist())  # T timesteps

        # we need to make a correlation matrix
        self.corr = jnp.array(
            [
                [self.distance_function(i, j) for i in self.countries]
                for j in self.countries
            ]
        )
        self.corr = (1.0 + jnp.sqrt(3.0) * self.corr) * jnp.exp(-jnp.sqrt(3.0) * self.corr)
        self.corr = pd.DataFrame(
            self.corr, index=self.countries, columns=self.countries
        )

        # pi = data[self.inflation_column].values * 100
        data["pi"] = data[self.inflation_column] * 100

        # Initial Particles
        # OLD UNIVARIATE: [tau, lnsetasq, lnsepsilonsq, delta1, delta2, delta3, delta4]
        # NEW MULTIVARIATE: [tau_{1,...,n}, lnsetasq_{1,...,n}, lnsepsilonsq_{1,...,n}, delta1_{1,...,n}, delta2_{1,...,n}, delta3_{1,...,n}, delta4_{1,...,n}]
        # in total: 7 * n dimensions

        X0 = jnp.zeros((self.num_particles, 7 * n))
        X0[:, 0:n] = jnp.random.normal(
            size=(self.num_particles, n),
            loc=self.init_tau,
            scale=self.vague_prior_tau_sigma,
        )
        X0[:, n : 2 * n] = jnp.random.normal(
            size=(self.num_particles, n),
            loc=self.init_lnsetasq,
            scale=self.vague_prior_lnsetasq_sigma,
        )
        X0[:, 2 * n : 3 * n] = jnp.random.normal(
            size=(self.num_particles, n),
            loc=self.init_lnsepsilonsq,
            scale=self.vague_prior_lnsepsilonsq_sigma,
        )

        # deltas
        for i in range(3, 7):
            X0[:, i * n : (i + 1) * n] = jnp.random.normal(
                size=(self.num_particles, n),
                loc=self.init_delta,
                scale=self.vague_prior_delta_sigma,
            )

        W0 = jnp.ones(self.num_particles) / self.num_particles

        # history of X's
        X = jnp.zeros((n + 1, self.num_particles, 7 * n))
        W = jnp.zeros((n + 1, self.num_particles))
        X[0, :, :] = X0
        W[0, :] = W0

        # Seasonality indicator function
        # first we determine which modulo corresponds to Q1
        t0_season = (self.times[0].month - 1) // 3

        def seas(i, t):
            """
            = 1 iff timestamp t corresponds to season i
            season 0: Q1
            ...
            season 3: Q4
            """

            return 1 if t % 4 == ((i - t0_season) % 4) else 0

        def update_single_particle(prev_x, t, i, n):
            x = jnp.zeros(7 * n)
            x[t, i, n : 2 * n] = jnp.random.normal(
                loc=X[t - 1, i, n : 2 * n], scale=jnp.sqrt(self.gamma)
            )

            # lnsepsilonsq
            X[t, i, 2 * n : 3 * n] = jnp.random.normal(
                loc=X[t - 1, i, 2 * n : 3 * n], scale=jnp.sqrt(self.gamma)
            )

            # deltas
            for delta_idx in [3, 4, 5, 6]:
                X[t, i, delta_idx * n : (delta_idx + 1) * n] = (
                    jnp.random.normal(
                        loc=X[t - 1, i, delta_idx * n : (delta_idx + 1) * n],
                        scale=jnp.sqrt(self.theta),
                    )
                    if seas(delta_idx - 3, t)
                    else X[t - 1, i, delta_idx * n : (delta_idx + 1) * n]
                )

            # restrict sum of deltas to be 0 per country
            for country_idx in range(n):
                idx = [
                    3 * n + country_idx,
                    4 * n + country_idx,
                    5 * n + country_idx,
                    6 * n + country_idx,
                ]
                X[t, i, idx] -= jnp.mean(X[t, i, idx])

            # add the epsilon noise
            # We calculate the covariance matrix for the multivariate normal distribution
            # by taking the outer product of the standard deviations with itself
            # and multiplying it element-wise with the correlation matrix.
            epsilon_cov = (
                self.corr.values
                * jnp.outer(
                    jnp.sqrt(
                        jnp.exp(X[t, i, 2 * n : 3 * n])
                    ),  # 2*n:3*n corresponds to lnsepsilonsq
                    jnp.sqrt(jnp.exp(X[t, i, 2 * n : 3 * n])),
                )
            )
            X[t, i, 0:n] = random.multivariate_normal(
                key = key,
                mean=X[t - 1, i, 0:n],
                cov=epsilon_cov,
            )


        # History of tau + delta_1 * seas_1 + ... + delta_4 * seas_4
        etauplusdeltas = []
        for t, corresponding_time in zip(range(1, len(self.times) + 1), self.times):
            print("a")
            # Step 1: predict and update
            W[t, :] = jnp.ones(self.num_particles) / self.num_particles
            for i in range(self.num_particles):
                # lnsetasq
                X[t, i, n : 2 * n] = jnp.random.normal(
                    loc=X[t - 1, i, n : 2 * n], scale=jnp.sqrt(self.gamma)
                )

                # lnsepsilonsq
                X[t, i, 2 * n : 3 * n] = jnp.random.normal(
                    loc=X[t - 1, i, 2 * n : 3 * n], scale=jnp.sqrt(self.gamma)
                )

                # deltas
                for delta_idx in [3, 4, 5, 6]:
                    X[t, i, delta_idx * n : (delta_idx + 1) * n] = (
                        jnp.random.normal(
                            loc=X[t - 1, i, delta_idx * n : (delta_idx + 1) * n],
                            scale=jnp.sqrt(self.theta),
                        )
                        if seas(delta_idx - 3, t)
                        else X[t - 1, i, delta_idx * n : (delta_idx + 1) * n]
                    )

                # restrict sum of deltas to be 0 per country
                for country_idx in range(n):
                    idx = [
                        3 * n + country_idx,
                        4 * n + country_idx,
                        5 * n + country_idx,
                        6 * n + country_idx,
                    ]
                    X[t, i, idx] -= jnp.mean(X[t, i, idx])

                # add the epsilon noise
                # We calculate the covariance matrix for the multivariate normal distribution
                # by taking the outer product of the standard deviations with itself
                # and multiplying it element-wise with the correlation matrix.
                epsilon_cov = (
                    self.corr.values
                    * jnp.outer(
                        jnp.sqrt(
                            jnp.exp(X[t, i, 2 * n : 3 * n])
                        ),  # 2*n:3*n corresponds to lnsepsilonsq
                        jnp.sqrt(jnp.exp(X[t, i, 2 * n : 3 * n])),
                    )
                )
                X[t, i, 0:n] = random.multivariate_normal(
                    key = key,
                    mean=X[t - 1, i, 0:n],
                    cov=epsilon_cov,
                )

            # Mean of pi is tau + delta_1 * seas_1 + ... + delta_4 * seas_4
            # T x n matrix
            mean_vals = (
                X[t, :, 0:n]
                + X[t, :, 3 * n : 4 * n] * seas(0, t)
                + X[t, :, 4 * n : 5 * n] * seas(1, t)
                + X[t, :, 5 * n : 6 * n] * seas(2, t)
                + X[t, :, 6 * n : 7 * n] * seas(3, t)
            )
            etauplusdeltas.append(jnp.mean(mean_vals, axis=0))
            scale_vals = jnp.sqrt(
                jnp.exp(X[t, :, n : 2 * n])
            )  # 1*n:2*n corresponds to lnsetasq
            # T x n as well

            current_timestep_data = data[data[self.date_column] == corresponding_time]
            current_country_idxs = [
                self.countries.index(country)
                for country in current_timestep_data["country"]
            ]

            # T x len(current_country_idxs)
            mean_vals = mean_vals[:, current_country_idxs]
            scale_vals = scale_vals[:, current_country_idxs] 

            # TODO: multivariate normal pdf
            W[t, :] *= jnp.prod(
                jnorm.pdf(
                    current_timestep_data["pi"].values,
                    loc=mean_vals,
                    scale=scale_vals,
                ),
                axis=1,
            )
            if jnp.sum(W[t, :]) == 0:
                print("WARNING: All weights are zero. Resampling will fail.")
                print(f"(t = {t}, corresponding_time = {corresponding_time})")
            W[t, :] = W[t, :] / jnp.sum(W[t, :])

            # Step 2: resample
            indices = jnp.random.choice(
                self.num_particles, size=self.num_particles, replace=True, p=W[t, :]
            )
            X[t, :, :] = X[t, indices, :]

        if self.aggregation_method == "median":
            out = pd.DataFrame(
                {
                    "date": self.times,
                    "etau": jnp.median(X[1:, :, 0:n], axis=1)
                    / 100,  # convert back to percentage
                    "etauplusdeltas": etauplusdeltas,  # TODO
                    "elnsetasq": jnp.median(X[1:, :, n : 2 * n], axis=1),  # OTHER SCALE!
                    "esigmaeta": jnp.median(
                        jnp.sqrt(jnp.exp(X[1:, :, n : 2 * n])), axis=1
                    ),
                    "elnsepsilonsq": jnp.median(X[1:, :, 2 * n : 3 * n], axis=1),
                    "esigmaepsilon": jnp.median(
                        jnp.sqrt(jnp.exp(X[1:, :, 2 * n : 3 * n])), axis=1
                    ),
                    "edelta1": jnp.median(X[1:, :, 3 * n : 4 * n], axis=1) / 100,
                    "edelta2": jnp.median(X[1:, :, 4 * n : 4 * n], axis=1) / 100,
                    "edelta3": jnp.median(X[1:, :, 5 * n : 6 * n], axis=1) / 100,
                    "edelta4": jnp.median(X[1:, :, 6 * n : 7 * n], axis=1) / 100,
                    "meff": 1 / jnp.sum(W[1:, :] ** 2, axis=1),
                    # "inflation": data["inflation"].values,
                    "inflation": data.groupby(self.date_column)[self.inflation_column]
                    .median()
                    .unstack()
                    .reindex(self.times, axis=0)
                    .reindex(self.countries, axis=1)
                    .values,
                    "country": jnp.repeat(self.countries, len(self.times)),
                }
            )
        elif self.aggregation_method == "distribution":
            raise NotImplementedError(
                "Distribution aggregation not implemented yet for MUCSVSS model."
            )

            def getPDFRow(row):
                pdf = gaussian_kde(row)
                return pdf

            out = pd.DataFrame(
                {
                    "date": data["date"].values,
                    "etau": np.apply_along_axis(
                        getPDFRow, axis=1, arr=(X[1:, :, 0] / 100)
                    ),  # convert back to percentage
                    "etauplusdeltas": etauplusdeltas,  # TODO
                    "elnsetasq": np.apply_along_axis(
                        getPDFRow, axis=1, arr=(X[1:, :, 1])
                    ),  # OTHER SCALE!
                    "esigmaeta": np.apply_along_axis(
                        getPDFRow, axis=1, arr=(np.sqrt(np.exp(X[1:, :, 1])))
                    ),
                    "elnsepsilonsq": np.apply_along_axis(
                        getPDFRow, axis=1, arr=(X[1:, :, 2])
                    ),
                    "esigmaepsilon": np.apply_along_axis(
                        getPDFRow, axis=1, arr=(np.sqrt(np.exp(X[1:, :, 2])))
                    ),
                    "edelta1": np.apply_along_axis(
                        getPDFRow, axis=1, arr=(X[1:, :, 3] / 100)
                    ),
                    "edelta2": np.apply_along_axis(
                        getPDFRow, axis=1, arr=(X[1:, :, 4] / 100)
                    ),
                    "edelta3": np.apply_along_axis(
                        getPDFRow, axis=1, arr=(X[1:, :, 5] / 100)
                    ),
                    "edelta4": np.apply_along_axis(
                        getPDFRow, axis=1, arr=(X[1:, :, 6] / 100)
                    ),
                    "meff": 1 / np.sum(W[1:, :] ** 2, axis=1),
                    "inflation": data["inflation"].values,
                }
            )

        #out[self.country_column] = data[self.country_column].iloc[0]

        return out.set_index([self.country_column, self.date_column])

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict the state at time t.

        E(pi_{t+1} | I_t)
        = E(tau_{t+1} + delta_{1,t} * seas_1(t+1) + ... + delta_{4,t} * seas_4(t+1) | I_t) for correct i
        = tau_t + E(delta_{1,t+1} | I_t) * seas_1(t + 1) + ... + E(delta_{4,t+1} | I_t) * seas_4(t + 1)
        = tau_t + delta_{1,t} * seas_1(t + 1) + ... + delta_{4,t} * seas_4(t + 1)
        """
        if not hasattr(self, "stored_state_means"):
            raise ValueError("Model has not been run_pf'd yet.")

        return pd.DataFrame(
            [
                self._predict(data.loc[data[self.country_column] == country])
                for country in data[self.country_column].unique()
            ]
        )

    def _predict(self, data: pd.DataFrame):
        """
        Predict the state at time t for a single country.
        """
        # We have to find the state corresponding to the last row of data
        row = self.stored_state_means.loc[
            data[self.country_column].iloc[-1], data[self.date_column].iloc[-1]
        ]
        tau_tminus1 = row["etau"]
        delta_tminus1 = row[["edelta1", "edelta2", "edelta3", "edelta4"]]

        # Seasonality indicator function
        # first we determine which modulo corresponds to Q1
        t0_season = (data[self.date_column].iloc[0].month - 1) // 3

        def seas(i, t):
            """
            = 1 iff timestamp t corresponds to season i
            season 0: Q1
            ...
            season 3: Q4
            """

            return 1 if t % 4 == ((i - t0_season) % 4) else 0

        tplus1 = len(data) + 1

        if self.aggregation_method == "distribution":
            minVal = min(
                min(tau_tminus1.dataset[0]),
                min(delta_tminus1["edelta1"].dataset[0]),
                min(delta_tminus1["edelta2"].dataset[0]),
                min(delta_tminus1["edelta3"].dataset[0]),
                min(delta_tminus1["edelta4"].dataset[0]),
            )
            maxVal = max(
                max(tau_tminus1.dataset[0]),
                max(delta_tminus1["edelta1"].dataset[0]),
                max(delta_tminus1["edelta2"].dataset[0]),
                max(delta_tminus1["edelta3"].dataset[0]),
                max(delta_tminus1["edelta4"].dataset[0]),
            )
            vals = np.linspace(minVal, maxVal, 1000)

            pdf_tau_tminus1 = tau_tminus1(vals)
            if seas(0, tplus1) == 1:
                pdf_edelta = delta_tminus1["edelta1"](vals)
            elif seas(1, tplus1) == 1:
                pdf_edelta = delta_tminus1["edelta2"](vals)
            elif (seas(2, tplus1)) == 1:
                pdf_edelta = delta_tminus1["edelta3"](vals)
            elif (seas(3, tplus1)) == 1:
                pdf_edelta = delta_tminus1["edelta4"](vals)

            # inflation = np.convolve(pdf_tau_tminus1,pdf_edelta, mode='same')
            inflation = fftconvolve(pdf_tau_tminus1, pdf_edelta)
            vals_convolved = np.linspace(vals[0], vals[-1], len(inflation))

            area = simps(inflation)
            normalized_inflation = inflation / area
            return {
                "inflation": {
                    "pdf": normalized_inflation,
                    "inflation_grid": vals_convolved,
                },
                "country": data["country"].iloc[0],
                "date": data["date"].iloc[-1] + pd.DateOffset(months=3),
            }

        elif self.aggregation_method == "median":
            return {
                "inflation": (
                    tau_tminus1
                    + delta_tminus1["edelta1"] * seas(0, tplus1)
                    + delta_tminus1["edelta2"] * seas(1, tplus1)
                    + delta_tminus1["edelta3"] * seas(2, tplus1)
                    + delta_tminus1["edelta4"] * seas(3, tplus1)
                ),
                "country": data["country"].iloc[0],
                "date": data["date"].iloc[-1] + pd.DateOffset(months=3),
            }
