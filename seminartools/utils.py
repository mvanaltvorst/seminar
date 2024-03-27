from functools import lru_cache
import math
import pandas as pd
import pycountry as pyc
from seminartools.data import read_gdp

@lru_cache(maxsize=None)
def geo_distance(countryA: str, countryB: str) -> float:
    """
    Calculate the geographical distance between two countries.

    Parameters:
    - countryA (str): The name of the first country.
    - countryB (str): The name of the second country.

    Returns:
    - float: The geographical distance between the two countries in kilometers.
    """
    filepath = "./../../assets/dist_cepii.xls"
    df_distance = pd.read_excel(filepath)
    countryA = pyc.countries.get(name=countryA)
    countryB = pyc.countries.get(name=countryB)

    # Use the ISO alpha-3 country codes to find the distance.
    row = df_distance[(df_distance['iso_o'] == countryA.alpha_3) & (df_distance['iso_d'] == countryB.alpha_3)]
    dist = row['dist'].item()
    
    return dist

@lru_cache(maxsize=None)
def gdp(country: str, year: int, quarter: int) -> float:
    """
    Retrieve the GDP of a given country for a specified year and quarter.

    Parameters:
    - country (str): The name of the country.
    - year (int): The year of interest.
    - quarter (int): The quarter of interest.

    Returns:
    - float: The GDP of the country for the specified time period.
    """
    df = read_gdp()
    # Convert quarter to the starting month of the quarter.
    month = quarter * 3 - 2
    time = f"{year}-{month:02d}-01"
    return df.loc[time, country]

def gravityTradeDistance(countryA: str, countryB: str, year: int, quarter: int) -> float:
    """
    Estimate trade potential between two countries using a simple gravity model of trade,
    which involves the countries' GDPs and the geographical distance between them.

    Parameters:
    - countryA (str): The name of the first country.
    - countryB (str): The name of the second country.
    - year (int): The year of interest.
    - quarter (int): The quarter of interest.

    Returns:
    - float: The calculated trade potential between the two countries.
    """
    GDP_A = gdp(countryA, year, quarter)
    GDP_B = gdp(countryB, year, quarter)
    dist = geo_distance(countryA, countryB)
    
    return GDP_A * GDP_B / dist

def proposalDistance(countryA: str, countryB: str, year: int, quarter: int) -> float:
    """
    Calculate a proposed distance metric between two countries, incorporating economic and geographical factors.

    Parameters:
    - countryA (str): The name of the first country.
    - countryB (str): The name of the second country.
    - year (int): The year of interest.
    - quarter (int): The quarter of interest.

    Returns:
    - float: The proposed distance metric.
    """
    theta = 0  # Placeholder for future use, indicating an adjustment factor.
    gamma = 0  # Not used in the current formula but could be incorporated later.
    dist = geo_distance(countryA, countryB) + theta
    # Adjust distance calculation based on GDP differences.
    return dist / (math.log(gdp(countryA, year, quarter)) - math.log(gdp(countryB, year, quarter)))**2
