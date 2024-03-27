import pandas as pd
from .base_model import BaseModel


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
