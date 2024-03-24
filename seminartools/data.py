import pandas as pd
import numpy as np


def read_inflation(
    *,
    filepath="./../../assets/Inflation-data.xlsx",
    sheet_name="hcpi_q",
    drop_non_complete_countries=True,
    first_difference=True,
    drop_countries=["Iceland", "Colombia", "Indonesia"],
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
    df_hcpi["yearmonth"] = pd.to_datetime(df_hcpi["yearmonth"])

    # if hcpi is 0, replace with NaN
    df_hcpi.loc[df_hcpi["hcpi"].abs() < 1e-6, "hcpi"] = np.nan

    # Drop countries with weird data
    for country in drop_countries:
        df_hcpi = df_hcpi[df_hcpi["Country"] != country]

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
        df_hcpi_pct_change = df_hcpi_pct_change.rename(columns={"hcpi": "inflation"})
        # drop na's
        df_hcpi_pct_change = df_hcpi_pct_change.dropna()
        return df_hcpi_pct_change
    else:
        return df_hcpi


def read_commodity(
    *,
    filepath="./../../assets/CMO-Historical-Data-Monthly.xlsx",
    sheet_name_1="Monthly Prices",
    sheet_name_2="Monthly Indices",
    header_table_1=6,
    header_table_2=9,
    first_difference=True,
    relevant_variables=['CRUDE_PETRO', 'iNATGAS', 'iAGRICULTURE', 'iMETMIN', 'iPRECIOUSMET']
):
    """
    Reads commodity price data from the CMO-Historical-Data-Monthly.xlsx file and returns a dataframe.

    Parameters
    ----------
    filepath : str
        Path to the excel file.
    sheet_name : str
        Name of the sheet in the excel file.
    header_table_1 : int
        Index of the header of the first table.
    header_table_2 : int
        Index of the header of the second table.
    first_difference : bool
        If True, compute the first difference of the commodity prices.
        if False, return the commodity prices as is.
    relevant_variables : list[str]
        List of relevant variables to keep in the dataframe.
    """
    df_1 = pd.read_excel(filepath, sheet_name=sheet_name_1, header=header_table_1).set_index("Unnamed: 0", drop = True)
    df_2 = pd.read_excel(filepath, sheet_name=sheet_name_2, header=header_table_2).set_index("Unnamed: 0", drop = True)
    df = pd.concat([df_1, df_2], axis=1)[relevant_variables] 
    df = df.apply(pd.to_numeric, errors='coerce').dropna().rename_axis("Date")
    df.index = pd.to_datetime(df.index, format = "%YM%m")
    df = df.resample('Q').mean()
    if first_difference:
        df = df.pct_change().dropna()
    return df




