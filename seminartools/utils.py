from functools import lru_cache
import math
import pandas as pd
import pycountry as pyc
from .data import read_gdp_level
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pycountry_convert as pc

def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name

# # Example
# country_name = 'Germany'
# print(country_to_continent(country_name))

# We cache the distance matrix
# because it makes the geo_distance method quicker
DF_DISTANCE_FILEPATH = "./../../assets/dist_cepii.xls"
DF_DISTANCE = pd.read_excel(DF_DISTANCE_FILEPATH)


def geo_distance(
    countryA: str,
    countryB: str,
    country_remap: dict[str, str] = {"Korea": "Korea, Republic of"},
    use_geopy = False
) -> float:
    """
    Calculate the geographical distance between two countries.

    Parameters:
    - countryA (str): The name of the first country.
    - countryB (str): The name of the second country.
    - use_geopy: wheter to use the geopy package. This calculates Vincenty distance
        usefull for geopandas plotting

    Returns:
    - float: The geographical distance between the two countries in kilometers.
    """
    countryA_remapped = country_remap.get(countryA, countryA)
    countryB_remapped = country_remap.get(countryB, countryB)
    if countryA_remapped == countryB_remapped:
        return 0 # ensures we fulfill the axioms of a metric space

    

    if use_geopy:
        loc = Nominatim(user_agent="Geopy Library")

        # entering the location name
        getLocA = loc.geocode(countryA)
        getLocB = loc.geocode(countryB)
        return geodesic((getLocA.latitude, getLocA.longitude), (getLocB.latitude, getLocB.longitude)).km
    else: 
        countryA = pyc.countries.get(name=countryA_remapped)
        if countryA is None:
            raise ValueError(f"Country {countryA_remapped} not found.")
        countryB = pyc.countries.get(name=countryB_remapped)
        if countryB is None:
            raise ValueError(f"Country {countryB_remapped} not found.")
        
        # if two countries are in different continents, return a large distance
        if country_to_continent(countryA.name) != country_to_continent(countryB.name):
            return 10**8

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

import geopandas as gpd
    
def get_plot(
    data: pd.DataFrame,
    country_column: str = 'country',
    europe = False,
    coefficient_name: list = ["coefficient"],
    legend = True,
    include_missing = True

):
    """
    Creates a plot of coefficients on a world map
    Args:
    -data: a dataframe that contains a column with country names and a column with values 
    -country_column: names of columns that contain coefficients
    -europe: whether to plot europe or the entire world
    -legend: whether to include a legend
    -coefficient_name: the name of the coefficient column
    -include_missing: if True, missing countries will be grey, else they will be excluded
    """

    #get data of the world from geopandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world.rename(columns = {'name': country_column}) 
    if europe: 
        world = world[(world.continent == 'Europe')]
        #Drop Russia from europe data cause it looks ugly cause looks ugly
        world = world[world['country'] != 'Russia']
   
    world_merged = world.merge(data, how='left', left_on= country_column, right_on = country_column)
    
    #return plots of all coefficients 
    if include_missing:
        return [world_merged.plot(column= col, cmap='OrRd', legend=legend, legend_kwds={"shrink":.5},figsize=(30,20), missing_kwds= dict(color = "lightgrey",)) for col in coefficient_name]
    else:
        return [world_merged.plot(column= col, cmap='OrRd', legend=legend, legend_kwds={"shrink":.5},figsize=(30,20)) for col in coefficient_name]



  


   