import scipy
import numpy as np
from .base_model import BaseModel
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import ecdf

# num cpu
from multiprocessing import cpu_count

# nu follows a normal distribution with variance gamma
GAMMA = 0.2

# zeta follows a normal distribution with variance theta
THETA = 0.002

INIT_LNSETASQ = 0
INIT_LNSEPSILONSQ = 0
INIT_DELTA = 0

VAGUE_PRIOR_LNSETASQ_SIGMA = 3
VAGUE_PRIOR_LNSEPSILONSQ_SIGMA = 3
VAGUE_PRIOR_DELTA_SIGMA = 1
VAGUE_PRIOR_TAU_SIGMA = 1

N_CORES = cpu_count()


# Unobserved Component Stochastic Volatility Stochastic Seasonality Particle Filter
class UCSVSSModel(BaseModel):
    # We benefit from fitting on the entire dataset before predicting
    # Uses intermediate states to quickly evaluate the model on new data
    REQUIRES_ANTE_FULL_FIT = True

    def __init__(self, num_particles: int, stochastic_seasonality: bool):
        self.num_particles = num_particles
        self.country_column = "country"
        self.aggregation_method = None
        self.gamma = GAMMA

        # No stochastic seasonality is equivalent to theta = 0
        if stochastic_seasonality:
            self.theta = THETA
        else:
            self.theta = 0

        self.init_lnsetasq = INIT_LNSETASQ
        self.init_lnsepsilonsq = INIT_LNSEPSILONSQ
        self.init_delta = INIT_DELTA

        self.vague_prior_lnsetasq_sigma = VAGUE_PRIOR_LNSETASQ_SIGMA
        self.vague_prior_lnsepsilonsq_sigma = VAGUE_PRIOR_LNSEPSILONSQ_SIGMA
        if stochastic_seasonality:
            self.vague_prior_delta_sigma = VAGUE_PRIOR_DELTA_SIGMA
        else:
            self.vague_prior_delta_sigma = 0
        self.vague_prior_tau_sigma = VAGUE_PRIOR_TAU_SIGMA

    def fit(self, data: pd.DataFrame):
        """
        This model is not meant to be fitted to data every iteration using this method.
        Preferably, fit once on the entire dataset using `run_pf` and then
        figure out the historical predictions using the historical particles.
        """
        pass

    def run_pf(self, data: pd.DataFrame, aggregation_method : str = "median"):
        """
        Run the particle filter on the data of a single country.
        """
        self.aggregation_method = aggregation_method

        # dfs = data.groupby("Country").apply(self._run_pf)
        dfs = Parallel(n_jobs=N_CORES)(
            delayed(self._run_pf)(data.loc[data["country"] == country])
            for country in tqdm(data["country"].unique())
        )
        self.stored_state_means = pd.concat(dfs, axis=0)

    def _run_pf(self, data: pd.DataFrame):
        """
        Run the particle filter on the data of a single country.
        """
        n = len(data)
        pi = data["inflation"].values * 100

        # Initial Particles
        # [tau, lnsetasq, lnsepsilonsq, delta1, delta2, delta3, delta4]
        X0 = np.zeros((self.num_particles, 7))
        X0[:, 0] = np.random.normal(
            size=self.num_particles, loc=1, scale=self.vague_prior_tau_sigma
        )
        X0[:, 1] = np.random.normal(
            size=self.num_particles,
            loc=self.init_lnsetasq,
            scale=self.vague_prior_lnsetasq_sigma,
        )
        X0[:, 2] = np.random.normal(
            size=self.num_particles,
            loc=self.init_lnsepsilonsq,
            scale=self.vague_prior_lnsepsilonsq_sigma,
        )

        # deltas
        for i in range(3, 7):
            X0[:, i] = np.random.normal(
                size=self.num_particles,
                loc=self.init_delta,
                scale=self.vague_prior_delta_sigma,
            )

        W0 = np.ones(self.num_particles) / self.num_particles

        # history of X's
        X = np.zeros((n + 1, self.num_particles, 7))
        W = np.zeros((n + 1, self.num_particles))
        X[0, :, :] = X0
        W[0, :] = W0

        # Seasonality indicator function
        # first we determine which modulo corresponds to Q1
        t0_season = (data["date"].iloc[0].month - 1) // 3

        def seas(i, t):
            """
            = 1 iff timestamp t corresponds to season i
            season 0: Q1
            ...
            season 3: Q4
            """

            return 1 if t % 4 == ((i - t0_season) % 4) else 0

        # History of tau + delta_1 * seas_1 + ... + delta_4 * seas_4
        etauplusdeltas = []
        for t in range(1, n + 1):
            # Step 1: predict and update
            W[t, :] = np.ones(self.num_particles) / self.num_particles
            for i in range(self.num_particles):
                X[t, i, 1] = np.random.normal(
                    loc=X[t - 1, i, 1], scale=np.sqrt(self.gamma)
                )
                X[t, i, 2] = np.random.normal(
                    loc=X[t - 1, i, 2], scale=np.sqrt(self.gamma)
                )
                X[t, i, 3] = (
                    np.random.normal(loc=X[t - 1, i, 3], scale=np.sqrt(self.theta))
                    if seas(0, t)
                    else X[t - 1, i, 3]
                )
                X[t, i, 4] = (
                    np.random.normal(loc=X[t - 1, i, 4], scale=np.sqrt(self.theta))
                    if seas(1, t)
                    else X[t - 1, i, 4]
                )
                X[t, i, 5] = (
                    np.random.normal(loc=X[t - 1, i, 5], scale=np.sqrt(self.theta))
                    if seas(2, t)
                    else X[t - 1, i, 5]
                )
                X[t, i, 6] = (
                    np.random.normal(loc=X[t - 1, i, 6], scale=np.sqrt(self.theta))
                    if seas(3, t)
                    else X[t - 1, i, 6]
                )
                # restrict sum of deltas to be 0
                delta_mean = np.mean(X[t, i, 3:7])
                X[t, i, 3:7] -= delta_mean

                # add the epsilon noise
                X[t, i, 0] = np.random.normal(
                    loc=X[t - 1, i, 0], scale=np.sqrt(np.exp(X[t, i, 2]))
                )

            # Mean of pi is tau + delta_1 * seas_1 + ... + delta_4 * seas_4
            mean_vals = (
                X[t, :, 0]
                + X[t, :, 3] * seas(0, t)
                + X[t, :, 4] * seas(1, t)
                + X[t, :, 5] * seas(2, t)
                + X[t, :, 6] * seas(3, t)
            )
            etauplusdeltas.append(np.mean(mean_vals))
            scale_vals = np.sqrt(np.exp(X[t, :, 1]))

            W[t, :] *= scipy.stats.norm.pdf(
                pi[t - 1],
                loc=mean_vals,
                scale=scale_vals,
            )
            if np.sum(W[t, :]) == 0:
                print("WARNING: All weights are zero. Resampling will fail.")
                print("(country: {})".format(data["country"].iloc[0]))
                print(f"(t = {t})")
            W[t, :] = W[t, :] / np.sum(W[t, :])

            # Step 2: resample
            indices = np.random.choice(
                self.num_particles, size=self.num_particles, replace=True, p=W[t, :]
            )
            X[t, :, :] = X[t, indices, :]

        """out = pd.DataFrame(
            {
                "date": data["date"].values,
                "etau": X[1:, :, 0].mean(axis=1) / 100,  # convert back to percentage
                "etauplusdeltas": etauplusdeltas,  # TODO
                "elnsetasq": X[1:, :, 1].mean(axis=1),  # OTHER SCALE!
                "esigmaeta": np.sqrt(np.exp(X[1:, :, 1])).mean(axis=1),
                "elnsepsilonsq": X[1:, :, 2].mean(axis=1),
                "esigmaepsilon": np.sqrt(np.exp(X[1:, :, 2])).mean(axis=1),
                "edelta1": X[1:, :, 3].mean(axis=1) / 100,
                "edelta2": X[1:, :, 4].mean(axis=1) / 100,
                "edelta3": X[1:, :, 5].mean(axis=1) / 100,
                "edelta4": X[1:, :, 6].mean(axis=1) / 100,
                "meff": 1 / np.sum(W[1:, :] ** 2, axis=1),
                "inflation": data["inflation"].values,
            }
        )"""

        if self.aggregation_method == "median":
            out = pd.DataFrame(
                {
                    "date": data["date"].values,
                    "etau": np.median(X[1:, :, 0],axis=1) / 100,  # convert back to percentage
                    "etauplusdeltas": etauplusdeltas,  # TODO
                    "elnsetasq": np.median(X[1:, :, 1],axis=1),  # OTHER SCALE!
                    "esigmaeta": np.median(np.sqrt(np.exp(X[1:, :, 1])),axis=1),
                    "elnsepsilonsq": np.median(X[1:, :, 2],axis=1),
                    "esigmaepsilon": np.median(np.sqrt(np.exp(X[1:, :, 2])),axis=1),
                    "edelta1": np.median(X[1:, :, 3],axis=1) / 100,
                    "edelta2": np.median(X[1:, :, 4],axis=1) / 100,
                    "edelta3": np.median(X[1:, :, 5],axis=1) / 100,
                    "edelta4": np.median(X[1:, :, 6],axis=1) / 100,
                    "meff": 1 / np.sum(W[1:, :] ** 2, axis=1),
                    "inflation": data["inflation"].values,
                }
            )
        elif self.aggregation_method == "distribution":
            def getECDFRow(row):
                return ecdf(row).cdf
            print(np.apply_along_axis(getECDFRow, axis=1,arr=(X[1:, :, 0] / 100)))
            out = pd.DataFrame(
                {
                    "date": data["date"].values,
                    "etau": np.apply_along_axis(getECDFRow, axis=1,arr=(X[1:, :, 0] / 100)),  # convert back to percentage
                    "etauplusdeltas": etauplusdeltas,  # TODO
                    "elnsetasq": np.apply_along_axis(getECDFRow, axis = 1, arr=X[1:, :, 1]),  # OTHER SCALE!
                    "esigmaeta": np.apply_along_axis(getECDFRow,axis=1,arr=np.sqrt(np.exp(X[1:, :, 1]))),
                    "elnsepsilonsq": np.apply_along_axis(getECDFRow, axis = 1,arr=X[1:, :, 2]),
                    "esigmaepsilon": np.apply_along_axis(getECDFRow, axis = 1, arr=np.sqrt(np.exp(X[1:, :, 2]))),
                    "edelta1": np.apply_along_axis(getECDFRow, axis = 1, arr = (X[1:, :, 3] / 100)),
                    "edelta2": np.apply_along_axis(getECDFRow, axis = 1, arr = (X[1:, :, 4] / 100)),
                    "edelta3": np.apply_along_axis(getECDFRow, axis = 1, arr= (X[1:, :, 5] / 100)),
                    "edelta4": np.apply_along_axis(getECDFRow, axis = 1, arr= (X[1:, :, 6] / 100)),
                    "meff": 1 / np.sum(W[1:, :] ** 2, axis=1),
                    "inflation": data["inflation"].values,
                }
            )
        print(self.aggregation_method)
        print(out)
        out["country"] = data["country"].iloc[0]

        return out.set_index(["country", "date"])

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
                self._predict(data.loc[data["country"] == country])
                for country in data["country"].unique()
            ]
        )

    def _predict(self, data: pd.DataFrame):
        """
        Predict the state at time t for a single country.
        """
        # We have to find the state corresponding to the last row of data
        row = self.stored_state_means.loc[
            data["country"].iloc[-1], data["date"].iloc[-1]
        ]
        tau_tminus1 = row["etau"]
        delta_tminus1 = row[["edelta1", "edelta2", "edelta3", "edelta4"]]

        # Seasonality indicator function
        # first we determine which modulo corresponds to Q1
        t0_season = (data["date"].iloc[0].month - 1) // 3

        def seas(i, t):
            """
            = 1 iff timestamp t corresponds to season i
            season 0: Q1
            ...
            season 3: Q4
            """

            return 1 if t % 4 == ((i - t0_season) % 4) else 0

        tplus1 = len(data) + 1

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
