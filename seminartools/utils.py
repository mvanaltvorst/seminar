from functools import lru_cache
import math
import pandas as pd
import pycountry as pyc
from .data import read_gdp_level


# We cache the distance matrix
# because it makes the geo_distance method quicker
DF_DISTANCE_FILEPATH = "./../../assets/dist_cepii.xls"
DF_DISTANCE = pd.read_excel(DF_DISTANCE_FILEPATH)


def geo_distance(
    countryA: str,
    countryB: str,
    country_remap: dict[str, str] = {"Korea": "Korea, Republic of"},
) -> float:
    """
    Calculate the geographical distance between two countries.

    Parameters:
    - countryA (str): The name of the first country.
    - countryB (str): The name of the second country.

    Returns:
    - float: The geographical distance between the two countries in kilometers.
    """
    countryA_remapped = country_remap.get(countryA, countryA)
    countryB_remapped = country_remap.get(countryB, countryB)
    if countryA_remapped == countryB_remapped:
        return 0 # ensures we fulfill the axioms of a metric space

    countryA = pyc.countries.get(name=countryA_remapped)
    if countryA is None:
        raise ValueError(f"Country {countryA_remapped} not found.")
    countryB = pyc.countries.get(name=countryB_remapped)
    if countryB is None:
        raise ValueError(f"Country {countryB_remapped} not found.")

    # Use the ISO alpha-3 country codes to find the distance.
    row = DF_DISTANCE[
        (DF_DISTANCE["iso_o"] == countryA.alpha_3)
        & (DF_DISTANCE["iso_d"] == countryB.alpha_3)
    ]
    dist = row["dist"].item()

    return dist


GDP_LEVEL_DF = read_gdp_level()

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
    # Convert quarter to the starting month of the quarter.
    month = quarter * 3 - 2
    time = f"{year}-{month:02d}-01"
    return GDP_LEVEL_DF.loc[time, country]


def gravity_trade_distance(
    countryA: str, countryB: str, year: int, quarter: int
) -> float:
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


def proposal_distance(countryA: str, countryB: str, year: int, quarter: int) -> float:
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
    return (
        dist
        / (
            math.log(gdp(countryA, year, quarter))
            - math.log(gdp(countryB, year, quarter))
        )
        ** 2
    )
