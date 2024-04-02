import pandas as pd
from .base_model import BaseModel
from ..time_series_split import TimeSeriesSplit
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm


def h_period_ahead_forecast(
    model: BaseModel, data: pd.DataFrame, start_date: str, h: int = 1
):
    """
    Performs recursive forecasting for h periods ahead for a given model.
    Handles exogenous variables.
    We return a DataFrame with the following columns:
    - yearmonth: The date of the forecast.
    - Country: The country for which the forecast was made.
    - ... (other columns in the data DataFrame excluding inflation and country_column)
    - inflation: The predicted inflation rate.

    Start date is the first date to forecast.

    Args:

    """
    if model.REQUIRES_ANTE_FULL_FIT and h > 1:
        raise ValueError("""
You are trying to make a forecast for more than one period ahead, but the model requires a full fit.
Please set h to 1 or use a different model.
        """)

    # We need to start earlier than the start date because we forecast h periods ahead
    current_time = pd.to_datetime(start_date) - pd.DateOffset(months=3 * (h - 1))

    predictions = []
    while True:
        forecast_time = current_time
        auxiliary_df = data[data["date"] < forecast_time].copy()
        for i in range(h):
            preds = model.predict(auxiliary_df)

            # Add the predictions to the auxiliary dataframe
            auxiliary_df = pd.concat([auxiliary_df, preds], ignore_index=True)

            forecast_time += pd.DateOffset(months=3)

        predictions.append(preds)

        # We stop making forecasts if we have reached the end of the data
        if current_time + pd.DateOffset(months=3 * h) > data.iloc[-1]["date"]:
            break

        current_time += pd.DateOffset(months=3)

    return pd.concat(predictions, ignore_index=True)


def make_oos_predictions(
    model: BaseModel,
    data: pd.DataFrame,
    retrain_time_series_split: TimeSeriesSplit,
    h: int = 1,
    progress: bool = False,
    num_cores: int = 1,
):
    """
    Makes out-of-sample predictions for a given model.
    Handles exogenous variables.
    We return a DataFrame with the following columns:
    - yearmonth: The date of the forecast.
    - Country: The country for which the forecast was made.
    - ... (other columns in the data DataFrame excluding inflation and country_column)
    - inflation: The predicted inflation rate.

    Start date is the first date to forecast.

    Args:

    """
    # acc = []
    iterator = retrain_time_series_split.split(data)
    if progress:
        iterator = tqdm(
            iterator, total=retrain_time_series_split.num_splits, desc="Splits"
        )

    if model.REQUIRES_ANTE_FULL_FIT:
        print("Fitting model on the full dataset...")
        model.full_fit(data)
        print("Fitted!")

    def worker(train_df, test_df, test_start_date):
        model.fit(train_df)
        # We forecast h periods ahead for each test set
        predictions = h_period_ahead_forecast(model, test_df, test_start_date, h)
        return predictions

    # return pd.concat(acc, ignore_index=True)
    return pd.concat(
        Parallel(n_jobs=num_cores)(
            delayed(worker)(train_df, test_df, test_start_date)
            for train_df, test_df, test_start_date in iterator
        )
        if num_cores > 1
        else [
            worker(train_df, test_df, test_start_date)
            for train_df, test_df, test_start_date in iterator
        ],
        ignore_index=True,
    )


def _get_stats(
    original_data: pd.DataFrame,
    predictions: pd.DataFrame,
    date_column: str = "date",
    country_column: str = "country",
):
    """
    Calculates the mean squared error, mean absolute error, and mean absolute percentage error
    for the predictions.
    """
    merged = predictions.merge(
        original_data, on=[date_column, country_column], suffixes=("_pred", "_true")
    ).dropna()

    # do mincer zarnowitz and get intercept, slope, and r2
    merged["constant"] = 1
    model = sm.OLS(merged["inflation_true"], merged[["constant", "inflation_pred"]])
    results = model.fit()
    mz_intercept = results.params["constant"]
    mz_slope = results.params["inflation_pred"]
    mz_r2 = results.rsquared


    return pd.Series(
        {
            "mse": mean_squared_error(
                merged["inflation_true"], merged["inflation_pred"]
            ),
            "mae": mean_absolute_error(
                merged["inflation_true"], merged["inflation_pred"]
            ),
            "r2": r2_score(merged["inflation_true"], merged["inflation_pred"]),
            "mz_intercept": mz_intercept,
            "mz_slope": mz_slope,
            "mz_r2": mz_r2,
        }
    )


def get_stats(
    models: list[tuple[str, BaseModel]],
    data: pd.DataFrame,
    retrain_time_series_split: TimeSeriesSplit,
    h: int = 1,
    num_cores_parallel_models: int = 3,
    num_cores_parallel_splits: int = 5,
):
    """
    Compares the performance of multiple models on a given dataset.
    """

    def work_model(name, model):
        try:
            predictions = make_oos_predictions(
                model,
                data,
                retrain_time_series_split,
                h=h,
                progress=False,
                num_cores=num_cores_parallel_splits,
            )
        except Exception as e:
            print(e)
            # print stack trace
            import traceback
            traceback.print_exc()
            print(name)
            print(data["country"].unique())
            return None
        return _get_stats(data, predictions)

    return pd.DataFrame(
        Parallel(n_jobs=num_cores_parallel_models)(
            delayed(work_model)(name, model) for name, model in models
        ) if num_cores_parallel_models > 1 else [work_model(name, model) for name, model in models],
        index=[name for name, model in models],
    ).sort_values("mse")
