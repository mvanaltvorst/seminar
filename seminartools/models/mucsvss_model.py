from .base_model import BaseModel
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.integrate import simps
from scipy.signal import fftconvolve
from ..utils import geo_distance
import numpy as np
import jax.numpy as jnp
from jax import lax, random, jit, vmap, pmap, device_count
from jax.scipy.stats import norm
from tqdm import tqdm
import jax

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

        self.n_devices = device_count()
        self.n_particles_per_device = self.num_particles // self.n_devices
        assert (
            self.num_particles % self.n_devices == 0
        ), "Total particles must be divisible by the number of devices."

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

    def _construct_corr_matrix(self, countries: list[str]) -> pd.DataFrame:
        """
        Construct the correlation matrix based on the distance function.
        """
        # we need to make a correlation matrix
        corr = np.array(
            [
                [self.distance_function(i, j) for i in self.countries]
                for j in self.countries
            ]
        )
        corr = (1.0 + np.sqrt(3.0) * corr) * np.exp(-np.sqrt(3.0) * corr)
        corr = pd.DataFrame(corr, index=self.countries, columns=self.countries)
        return pd.DataFrame(
            [[self.distance_function(i, j) for i in countries] for j in countries],
            index=countries,
            columns=countries,
        )

    def _construct_initial_X_W(self, key: random.PRNGKey, n: int, T: int):
        """
        Construct the initial particles and weights.
        OLD UNIVARIATE: [tau, lnsetasq, lnsepsilonsq, delta1, delta2, delta3, delta4]
        NEW MULTIVARIATE: [tau_{1,...,n}, lnsetasq_{1,...,n}, lnsepsilonsq_{1,...,n}, delta1_{1,...,n}, delta2_{1,...,n}, delta3_{1,...,n}, delta4_{1,...,n}]
        in total: 7 * n dimensions
        """

        X0 = jnp.zeros((self.num_particles, 7 * n))
        key, subkey = random.split(key)
        X0 = X0.at[:, 0:n].set(
            self.init_tau
            + self.vague_prior_tau_sigma
            * random.normal(
                key=subkey,
                shape=(self.num_particles, n),
            )
        )
        key, subkey = random.split(key)
        X0 = X0.at[:, n : 2 * n].set(
            self.init_lnsetasq
            + self.vague_prior_lnsetasq_sigma
            * random.normal(
                key=subkey,
                shape=(self.num_particles, n),
            )
        )
        key, subkey = random.split(key)
        X0 = X0.at[:, 2 * n : 3 * n].set(
            self.init_lnsepsilonsq
            + self.vague_prior_lnsepsilonsq_sigma
            * random.normal(
                key=subkey,
                shape=(self.num_particles, n),
            )
        )

        # deltas
        for i in range(3, 7):
            key, subkey = random.split(key)
            X0 = X0.at[:, i * n : (i + 1) * n].set(
                self.init_delta
                + self.vague_prior_delta_sigma
                * random.normal(
                    key=subkey,
                    shape=(self.num_particles, n),
                )
            )

        # Weights in log space
        W0 = jnp.full(self.num_particles, -jnp.log(self.num_particles))

        # history of X's
        X = jnp.zeros((n + 1, self.num_particles, 7 * n))
        W = jnp.zeros((n + 1, self.num_particles))
        X = X.at[0, :, :].set(X0)
        W = W.at[0, :].set(W0)
        return X, W

    @staticmethod
    def _seas(i, t, t0_season):
        """
        = 1 iff timestamp t corresponds to season i
        season 0: Q1
        ...
        season 3: Q4
        """

        # return 1 if t % 4 == ((i - t0_season) % 4) else 0

        # Using lax.cond for JIT-compatible conditional logic
        return lax.cond(t % 4 == ((i - t0_season) % 4), lambda _: 1, lambda _: 0, None)

    @staticmethod
    def _update_deltas(delta_idx, prev_x, x, t, n, theta, t0_season, key):
        """
        Used to update the delta conditional on the seasonality indicator.
        """

        def true_fun(_):
            update = prev_x[delta_idx * n : (delta_idx + 1) * n] + jnp.sqrt(
                theta
            ) * random.normal(key=key, shape=(n,))
            return x.at[delta_idx * n : (delta_idx + 1) * n].set(update)

        def false_fun(_):
            return x  # No update needed, return x as is

        # Use lax.cond for JIT'able conditional execution
        return lax.cond(
            MUCSVSSModel._seas(i=delta_idx - 3, t=t, t0_season=t0_season),
            true_fun,
            false_fun,
            None,
        )

    @staticmethod
    def _update_single_particle(
        prev_x, t, n, gamma, theta, t0_season, corr_values, key
    ):
        """
        Update a single particle at time t using the previous particle at time t-1.
        Designed to be JIT'able.
        """
        x = jnp.zeros(7 * n)
        key, subkey = random.split(key)
        x = x.at[n : 2 * n].set(
            prev_x[n : 2 * n] + jnp.sqrt(gamma) * random.normal(key=subkey, shape=(n,))
        )

        # lnsepsilonsq
        key, subkey = random.split(key)
        x = x.at[2 * n : 3 * n].set(
            prev_x[2 * n : 3 * n]
            + jnp.sqrt(gamma) * random.normal(key=subkey, shape=(n,))
        )

        # deltas
        for delta_idx in [3, 4, 5, 6]:
            key, subkey = random.split(key)
            x = MUCSVSSModel._update_deltas(
                delta_idx=delta_idx,
                prev_x=prev_x,
                x=x,
                t=t,
                n=n,
                theta=theta,
                t0_season=t0_season,
                key=subkey,
            )

        # restrict sum of deltas to be 0 per country
        for country_idx in range(n):
            idx = jnp.array(
                [
                    3 * n + country_idx,
                    4 * n + country_idx,
                    5 * n + country_idx,
                    6 * n + country_idx,
                ]
            )
            # x[idx] -= jnp.mean(x[idx])
            x = x.at[idx].set(x[idx] - jnp.mean(x[idx]))

        # add the epsilon noise
        # We calculate the covariance matrix for the multivariate normal distribution
        # by taking the outer product of the standard deviations with itself
        # and multiplying it element-wise with the correlation matrix.
        epsilon_cov = corr_values * jnp.outer(
            # 2*n:3*n corresponds to lnsepsilonsq
            jnp.sqrt(jnp.exp(x[2 * n : 3 * n])),
            jnp.sqrt(jnp.exp(x[2 * n : 3 * n])),
        )

        # final use of key
        x = x.at[0:n].set(
            random.multivariate_normal(
                key=key,
                mean=prev_x[0:n],
                cov=epsilon_cov,
            )
        )
        return x

    def _run_pf(self, data: pd.DataFrame):
        """
        Run the particle filter on the data of a single country.
        """
        key = random.PRNGKey(42)

        self.countries = data[self.country_column].unique().tolist()  # n countries
        n = len(self.countries)
        self.times = sorted(data[self.date_column].unique().tolist())  # T timesteps

        self.corr = self._construct_corr_matrix(self.countries)

        # pi = data[self.inflation_column].values * 100
        data["pi"] = data[self.inflation_column] * 100

        key, subkey = random.split(key)
        X, W = self._construct_initial_X_W(subkey, n, len(self.times))
        # X: T x particles x 7*n
        # W: T x particles

        # Seasonality indicator function
        # we determine which modulo corresponds to Q1
        t0_season = (self.times[0].month - 1) // 3

        # update_particles = jit(
        #     vmap(
        #         update_single_particle,
        #         in_axes=(0, None, None, 0),
        #         out_axes=0,
        #     ),
        #     static_argnums=(2,),
        # )

        # jit_update_single_particle = jit(update_single_particle, static_argnums=(3,))
        # Wrapper function that uses vmap to vectorize update_single_particle over particles in a shard
        # We broadcast everything except the particles and keys
        update_particles_shard = vmap(
            MUCSVSSModel._update_single_particle,
            in_axes=(0, None, None, None, None, None, None, 0),
            out_axes=0,
        )
        # pmap to parallelize over devices
        # n, self.gamma, self.theta, t0_season, self.corr.values are always constant
        # but self.corr.values is not hashable
        update_particles_parallel = pmap(
            update_particles_shard,
            in_axes=(0, None, None, None, None, None, None, 0),
            static_broadcasted_argnums=(2, 3, 4, 5),
        )

        def get_XTminus1_subkeys_reshaped(
            X,
            t,
            key,
        ):
            """
            Update particles at time t and return the mean of tau + delta_1 * seas_1 + ... + delta_4 * seas_4
            """
            subkeys = random.split(key, self.num_particles)
            subkeys_reshaped = subkeys.reshape(
                self.n_devices, self.n_particles_per_device, -1
            )
            Xtminus1_reshaped = X[t - 1, :, :].reshape(
                self.n_devices, self.n_particles_per_device, -1
            )
            return Xtminus1_reshaped, subkeys_reshaped

        jit_get_XTminus1_subkeys_reshaped = jit(get_XTminus1_subkeys_reshaped)

        def update_X_W(X, W, X_updated, current_country_idxs, t, key):
            X = X.at[t, :, :].set(X_updated.reshape(self.num_particles, 7 * n))
            # Mean of pi is tau + delta_1 * seas_1 + ... + delta_4 * seas_4
            # T x n matrix
            mean_vals = (
                X[t, :, 0:n]
                + X[t, :, 3 * n : 4 * n] * MUCSVSSModel._seas(0, t, t0_season)
                + X[t, :, 4 * n : 5 * n] * MUCSVSSModel._seas(1, t, t0_season)
                + X[t, :, 5 * n : 6 * n] * MUCSVSSModel._seas(2, t, t0_season)
                + X[t, :, 6 * n : 7 * n] * MUCSVSSModel._seas(3, t, t0_season)
            )
            etauplusdeltas_element = jnp.mean(mean_vals, axis=0)
            scale_vals = jnp.sqrt(
                jnp.exp(X[t, :, n : 2 * n])
            )  # n:2*n corresponds to lnsetasq
            # dim T x n

            # T x len(current_country_idxs)
            mean_vals = mean_vals[:, current_country_idxs]
            scale_vals = scale_vals[:, current_country_idxs]
            #return X, etauplusdeltas_element, mean_vals, scale_vals

            # TODO: multivariate normal pdf
            W = W.at[t, :].set(
                W[t, :]
                + jnp.sum(
                    norm.logpdf(
                        current_timestep_data["pi"].values,
                        loc=mean_vals,
                        scale=scale_vals,
                    ),
                    axis=1,
                )
            )

            # For numerical stability, we subtract the max value from the log weights
            max_W = jnp.max(W[t, :])
            W = W.at[t, :].set(W[t, :] - max_W)
            W = W.at[t, :].set(jnp.exp(W[t, :]))
            W = W.at[t, :].set(W[t, :] / jnp.sum(W[t, :]))

            # if jnp.sum(W[t, :]) == 0:
            #     print("WARNING: All weights are zero. Resampling will fail.")
            #     print(f"(t = {t}, corresponding_time = {corresponding_time})")

            # Step 2: resample
            key, subkey = random.split(key)
            indices = random.choice(
                a=self.num_particles,
                shape=(self.num_particles,),
                replace=True,
                p=W[t, :],
                key=subkey,
            )

            # back to log space
            W = W.at[t, :].set(jnp.log(W[t, :]))

            X = X.at[t, :, :].set(X[t, indices, :])
            return X, W, etauplusdeltas_element

        jit_update_X_W = jit(update_X_W)

        # History of tau + delta_1 * seas_1 + ... + delta_4 * seas_4
        etauplusdeltas = []
        for t, corresponding_time in tqdm(
            zip(range(1, len(self.times) + 1), self.times), total=len(self.times)
        ):
            # Step 1: predict and update
            W = W.at[t, :].set(
                jnp.full(self.num_particles, -jnp.log(self.num_particles))
            )

            current_timestep_data = data[data[self.date_column] == corresponding_time]
            current_country_idxs = [
                self.countries.index(country)
                for country in current_timestep_data["country"]
            ]

            # Old: simple vmap update_particles
            # X = X.at[t, :, :].set(update_particles(X[t - 1, :, :], t, n, subkeys))

            key, subkey = random.split(key)

            # X, etauplusdeltas_element, mean_vals, scale_vals = (
            #     update_X_get_mean_scale(X, t, current_country_idxs, subkey)
            # )
            Xtminus1_reshaped, subkeys_reshaped = jit_get_XTminus1_subkeys_reshaped(
                X, t, subkey
            )
            X_updated = update_particles_parallel(
                Xtminus1_reshaped,
                t,
                n,
                self.gamma,
                self.theta,
                t0_season,
                self.corr.values,
                subkeys_reshaped,
            )
            key, subkey = random.split(key)

            X, W, etauplusdeltas_element = jit_update_X_W(X, W, X_updated, current_country_idxs, t, subkey)
            etauplusdeltas.append(etauplusdeltas_element)

        if self.aggregation_method == "median":
            out = pd.DataFrame(
                {
                    "date": self.times,
                    "etau": jnp.median(X[1:, :, 0:n], axis=1)
                    / 100,  # convert back to percentage
                    "etauplusdeltas": etauplusdeltas,  # TODO
                    "elnsetasq": jnp.median(
                        X[1:, :, n : 2 * n], axis=1
                    ),  # OTHER SCALE!
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
                    .values,
                    "country": jnp.repeat(jnp.array(self.countries), len(self.times)),
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

        # out[self.country_column] = data[self.country_column].iloc[0]

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
