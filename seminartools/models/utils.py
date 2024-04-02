import pandas as pd
from .base_model import BaseModel
from ..time_series_split import TimeSeriesSplit
from tqdm import tqdm
from joblib import Parallel, delayed


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
