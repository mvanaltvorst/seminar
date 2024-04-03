import pandas as pd
import numpy as np


def read_inflation(
    *,
    filepath="./../../assets/Inflation-data.xlsx",
    sheet_name="hcpi_q",
    drop_non_complete_countries=True,
    first_difference=True,
    drop_countries=["Iceland", "Colombia", "Indonesia"],
    mergeable_format=False,
    country_remaps={
        "Korea, Rep.": "Korea",
        "Turkey": "Türkiye",
    },
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

    column_of_importance = "hcpi"
    if first_difference:
        column_of_importance = "inflation"
        df_hcpi_pct_change = df_hcpi.sort_values("yearmonth")

        df_hcpi_pct_change["hcpi"] = df_hcpi_pct_change.groupby("Country")[
            "hcpi"
        ].pct_change()
        df_hcpi_pct_change = df_hcpi_pct_change.rename(columns={"hcpi": "inflation"})
        # drop na's
        df_hcpi_pct_change = df_hcpi_pct_change.dropna()
        df = df_hcpi_pct_change
    else:
        df = df_hcpi

    if mergeable_format:
        df = df[["Country", "yearmonth", column_of_importance]].rename(
            columns={"Country": "country", "yearmonth": "date"}
        )
        df = df.replace({"country": country_remaps})
        df = df.set_index(["country", "date"])

    return df


# TODO: date fucked up, not quarters
def read_commodity(
    *,
    filepath="./../../assets/CMO-Historical-Data-Monthly.xlsx",
    sheet_name_1="Monthly Prices",
    sheet_name_2="Monthly Indices",
    header_table_1=6,
    header_table_2=9,
    first_difference=True,
    relevant_variables=[
        "CRUDE_PETRO",
        "iNATGAS",
        "iAGRICULTURE",
        "iMETMIN",
        "iPRECIOUSMET",
    ],
    mergeable_format=False,
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
    df_1 = pd.read_excel(
        filepath, sheet_name=sheet_name_1, header=header_table_1
    ).set_index("Unnamed: 0", drop=True)
    df_2 = pd.read_excel(
        filepath, sheet_name=sheet_name_2, header=header_table_2
    ).set_index("Unnamed: 0", drop=True)
    df = pd.concat([df_1, df_2], axis=1)[relevant_variables]
    df = df.apply(pd.to_numeric, errors="coerce").dropna().rename_axis("Date")
    df.index = pd.to_datetime(df.index, format="%YM%m")

    if first_difference:
        df = df.pct_change().dropna()
        df = df.resample("Q").apply(lambda x: (1 + x).prod() - 1)  # quarterly return
    else:
        df = df.resample("Q").mean()  # mean of the quarter
    df.index = df.index - pd.tseries.offsets.QuarterBegin(startingMonth=1)

    if mergeable_format:
        # index: date
        # columns: commodity_{commodity_name}
        df = df.add_prefix("commodity_")
        df = df.rename_axis("date")

    return df


def read_gdp_growth(
    *,
    filepath="./../../assets/GDP-growth.xlsx",
    sheet_name="Quarterly real GDP growth",
    header=5,
    skipfooter=5,
    add_median=False,
    mergeable_format=False,
    country_remaps={"China (People's Republic of)": "China"},
):
    """
    Reads GDP growth data from the GDP-growth.xlsx file and returns a dataframe.
    This dataframe already contains first differences. There is no option
    to obtain the raw GDP index.

    Parameters
    ----------
    filepath : str
        Path to the excel file.
    sheet_name : str
        Name of the sheet in the excel file.
    header : int
        Index of the header of the table.
    skipfooter : int
        Number of rows to skip at the end of the table.
    add_median : bool
        If True, add a column with the median GDP growth rate across countries.
    """
    df = (
        pd.read_excel(
            filepath, sheet_name=sheet_name, header=header, skipfooter=skipfooter
        )
        .set_index("Period", drop=True)
        .drop(columns=["Unnamed: 1", "Unnamed: 2"])
    )

    # Rename columns: only half of the columns are named the respective time period.
    # We rename the columns to the time period of the previous column.

    for i in range(int(len(df.columns) / 2)):
        df.rename(columns={df.columns[2 * i + 1]: df.columns[2 * i]}, inplace=True)

    df = df.T
    df.drop(columns=["Country"], inplace=True)

    # Coerce non-numeric values to NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    # Delete rows that are all NaN
    df.dropna(how="all", inplace=True)

    # Change the index to datetime
    def parse_quarter_string(q_string):
        # Split the string into its components (e.g., "Q2-1947" -> ["Q2", "1947"])
        quarter_part, year_part = q_string.split("-")

        # Extract the quarter number
        quarter_number = int(quarter_part[1])  # Convert "2" to 2

        # Determine the month that corresponds to the quarter
        month = (
            quarter_number - 1
        ) * 3 + 1  # Quarter 1 starts in January, Quarter 2 in April, etc.

        # Create a datetime object for the first day of the starting month of the quarter
        return pd.Timestamp(year=int(year_part), month=month, day=1)

    # Apply the function to the index
    df.index = df.index.map(parse_quarter_string)

    if add_median:
        # Add median column
        df["median"] = df.median(axis=1)

    if mergeable_format:
        df = df.rename_axis("date")
        df = (
            df.stack()
            .reset_index()
            .rename(columns={"Period": "country", 0: "gdp_growth"})
        )
        df = df.replace({"country": country_remaps})
        df = df.set_index(["country", "date"])

    return df


def read_gdp_level(
    *,
    filepath="./../../assets/real-GDP.xlsx",
    sheet_name="level GDP",
    header=5,
    skipfooter=5,
    add_median=False,
    mergeable_format = False,
    country_remaps = {
        "China (People's Republic of)": "China"
    }
):
    """
    Reads GDP level data from the real-GDP.xlsx file and returns a dataframe.
    
    Parameters
    ----------
    filepath : str
        Path to the excel file.
    sheet_name : str
        Name of the sheet in the excel file.
    header : int
        Index of the header of the table.
    skipfooter : int
        Number of rows to skip at the end of the table.
    add_median : bool
        If True, add a column with the median GDP growth rate across countries.
    """

    df_gdp = pd.read_excel(filepath, sheet_name= sheet_name,header= header,skipfooter= skipfooter)

    #handle weird indent for last ten countries
    df_gdp.iloc[-10:,0]   = df_gdp.iloc[-10:,1]

    df_gdp.set_index('Period', inplace=True, drop=True)
    df_gdp.drop(columns=['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3'], inplace=True)
    
    # Rename columns: only half of the columns are named the respective time period.
    # We rename the columns to the time period of the previous column.
    for i in range(int(len(df_gdp.columns) / 2)):
        df_gdp.rename(
            columns={
                df_gdp.columns[2*i+1]: df_gdp.columns[2*i]
            },
            inplace=True
    )


    df_gdp = df_gdp.T
    df_gdp.drop(columns=['Country'], inplace=True)

    # Coerce non-numeric values to NaN
    df_gdp = df_gdp.apply(pd.to_numeric, errors='coerce')

    # Delete rows that are all NaN
    df_gdp.dropna(how='all', inplace=True)

    
    # Change the index to datetime
    def parse_quarter_string(q_string):
        # Split the string into its components (e.g., "Q2-1947" -> ["Q2", "1947"])
        quarter_part, year_part = q_string.split('-')
        
        # Extract the quarter number
        quarter_number = int(quarter_part[1])  # Convert "2" to 2
        
        # Determine the month that corresponds to the quarter
        month = (quarter_number - 1) * 3 + 1  # Quarter 1 starts in January, Quarter 2 in April, etc.
        
        # Create a datetime object for the first day of the starting month of the quarter
        return pd.Timestamp(year=int(year_part), month=month, day=1)

    # Apply the function to the index
    df_gdp.index = df_gdp.index.map(parse_quarter_string)


    if add_median:
        # Add median column
        df_gdp["median"] = df_gdp.median(axis=1)

    if mergeable_format:
        df_gdp = df_gdp.rename_axis("date")
        df_gdp = df_gdp.stack().reset_index().rename(columns = {
            "Period": "country",
            0: "gdp_growth"
        })
        df_gdp = df_gdp.replace({"country": country_remaps})
        df_gdp = df_gdp.set_index(["country", "date"])

    return df_gdp




# TODO: the time periods of this data are fucked up
# (quarters don't align)
def read_interest_rate(
    *,
    filepath="./../../assets/Monthly_interest_rates.xlsx",
    drop_columns=[
        "DATAFLOW_ID:Dataflow ID",
        "KEY:Timeseries Key",
        "FREQ:Frequency",
        "Unit",
        "Unit multiplier",
        "TIME_PERIOD:Period",
    ],
    mergeable_format=False,
):
    """
    Reads interest rate data from the Monthly_interest_rates.xlsx file and returns a dataframe.
    """
    df = pd.read_excel(filepath)
    df = df.drop(columns=drop_columns)
    df = df.set_index("REF_AREA:Reference area")
    df = df.T
    df.index.rename("Date", inplace=True)
    df.index = pd.to_datetime(df.index) - pd.offsets.MonthBegin(1)
    df = df.dropna(axis=1, how="all")  # Drop columns with all NaN values

    rows = df.index.get_loc("1999-01-01")
    euro_index = df.columns.get_loc("XM:Euro area")
    df_subset = df.iloc[rows:]

    # for all countries who only have missing values after 1999, i.e. the euro country areas
    # set their inflation after 1999-01-01 equal to that of euro area
    for country in df_subset.columns:
        if df_subset[country].isna().all():
            col_index = df.columns.get_loc(country)
            df.iloc[rows:, col_index] =  df.iloc[rows:, euro_index]


    # rename columns
    df.columns = [col.split(":")[1] for col in df.columns]
    # Quarterly average
    df = df.resample("Q").mean()  # Mean interest rate
    df.index = df.index - pd.tseries.offsets.QuarterBegin(startingMonth=1)

    if mergeable_format:
        df = (
            df.rename_axis("date")
            .stack()
            .reset_index()
            .rename(columns={"level_1": "country", 0: "interest_rate"})
            .set_index(["country", "date"])
        )

    return df


def read_unemployment(
    *,
    filepath="./../../assets/unemployment-ilo.csv",
    add_median=False,
    mergeable_format=False,
    country_remaps={
        "Korea, Republic of": "Korea",
        "Russian Federation": "Russia",
    },
):
    """
    Read unemployment data into a dataframe.

    Parameters
    ----------
    filepath : str
        Path to the csv file.
    add_median : bool
        If True, add a column with the median unemployment rate across countries.
    """
    df = pd.read_csv(filepath).drop(["Sex", "Age"], axis=1).set_index("Quarter")

    df.dropna(axis=1, how="all", inplace=True)  # Drop columns with all NaN values

    df.index = pd.PeriodIndex(
        df.index, freq="Q"
    ).to_timestamp()  # Convert the index to a datetime object

    df.sort_index(inplace=True)

    if add_median:
        df["median"] = df.median(axis=1)

    if mergeable_format:
        df = (
            df.rename_axis("date")
            .stack()
            .reset_index()
            .rename(columns={"level_1": "country", 0: "unemployment_rate"})
        )
        df = df.replace({"country": country_remaps})
        df = df.set_index(["country", "date"])

    return df


def read_merged(
        remove_countries = []

):
    """
    Args:
        remove_countries: A list of countries to be excluded from the dataset
    Returns a merged dataframe of all the data sources.
    """
    dfs = {
        "inflation": read_inflation(mergeable_format=True),
        "commodity": read_commodity(mergeable_format=True),
        "gdp_growth": read_gdp_growth(mergeable_format=True),
        "interest_rate": read_interest_rate(mergeable_format=True),
        "unemployment": read_unemployment(mergeable_format=True),
    }
    df = pd.concat([
        dfs['inflation'],
        #dfs['commodity'],
        dfs['gdp_growth'],
        dfs['interest_rate'],
        dfs['unemployment'],
    ], axis=1).join( # join level 1 of the multiindex with the commodity data
        dfs['commodity'],
        on='date',
    ).dropna()
    if len(remove_countries) != 0 :
        df = df.reset_index()
        select = df.country.apply(lambda x : x not in remove_countries)
        df = df[select].copy()
        df.set_index(['country', 'date'])
    return df
