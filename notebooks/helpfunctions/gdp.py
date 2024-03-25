from seminartools.data import read_gdp
import pandas as pd

def gdp(country: str, year: int, quarter: int) -> int:
    """
    Get the GDP of a country at a specified time

    """
    
    df = read_gdp()
    month = quarter * 3 - 2
    time = f"{year}-{month}-{0}{1}"
    return df.loc[time, country]

