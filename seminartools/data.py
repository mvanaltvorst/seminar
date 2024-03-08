import pandas as pd
import numpy as np


def read_inflation(
    *,
    filepath="./../../assets/Inflation-data.xlsx",
    sheet_name="hcpi_q",
    drop_non_complete_countries=True,
    first_difference=True,
):
    """
    Read the hcpi data from the world bank excel file and return a dataframe.

    Parameters
    ----------
    filepath : str
        Path to the excel file.
    sheet_name : str
        Name of the sheet in the excel file.
    drop_non_complete_countries : bool
        If True, drop countries with missing data.
    first_difference : bool
        If True, compute the first difference of the hcpi (quarterly inflation rate).
        if False, return the hcpi as is.
    """

    # world bank data, ignore last 2 rows
    df_hcpi = (
        pd.read_excel(
            filepath,
            sheet_name=sheet_name,
            skipfooter=2,
        )
        .drop(columns=["Indicator Type", "Series Name"])
        .set_index(["Country Code", "IMF Country Code", "Country"])
    )

    # identify unnamed column and remove all columns after it
    unnamed_col_idx = df_hcpi.columns.str.contains("Unnamed").argmax()
    df_hcpi = df_hcpi.iloc[:, :unnamed_col_idx]

    # turn columns into variables, set column name to "yearmonth"
    df_hcpi = (
        df_hcpi.stack()
        .reset_index()
        .rename(columns={"level_3": "yearmonth", 0: "hcpi"})
    )

    # yearmonth is 19861, 19862. Turn into datetime (1 is Q1, 2 is Q2, 3 is Q3, 4 is Q4)

    def into_datetime(yearmonth):
        yearmonth = str(yearmonth)
        year = int(yearmonth[:4])
        quarter = int(yearmonth[4])
        month = quarter * 3 - 2
        return f"{year}-{month:02d}"

    df_hcpi["yearmonth"] = df_hcpi["yearmonth"].apply(into_datetime)

    # if hcpi is 0, replace with NaN
    df_hcpi["hcpi"] = df_hcpi["hcpi"].replace(0, np.nan)
    df_hcpi = df_hcpi[df_hcpi["hcpi"].notna()]

    if drop_non_complete_countries:
        # We only retain the countries with 213 inflation data points (the maximum number of data points for a country)
        country_counts = df_hcpi.groupby("Country")["hcpi"].count()
        df_hcpi = df_hcpi[
            df_hcpi["Country"].isin(
                country_counts[country_counts == country_counts.max()].index
            )
        ]
        df_hcpi

    if first_difference:
        df_hcpi_pct_change = df_hcpi.sort_values("yearmonth")

        df_hcpi_pct_change["hcpi"] = df_hcpi_pct_change.groupby("Country")[
            "hcpi"
        ].pct_change()
        # drop na's
        df_hcpi_pct_change = df_hcpi_pct_change.dropna()
        return df_hcpi_pct_change
    else:
        return df_hcpi
