from gdp import gdp
import math
import pycountry as pyc
import pandas as pd


def geo_distance(countryA: str, countryB: str) -> int:
    """
    Get the distance between country A and B 

    """
    filepath="./../../assets/dist_cepii.xls"
    df_distance = pd.read_excel(filepath)
    countryA = pyc.countries.get(name= countryA)
    countryB = pyc.countries.get(name = countryB)

    # use the code of countries to get the geographical distance 
    row = df_distance[(df_distance['iso_o'] == countryA.alpha_3) & (df_distance['iso_d'] == countryB.alpha_3)]
    dist = row['dist'].item()
    
    return dist
     

def gravityTradeDistance(countryA: str, countryB: str, year: int, quarter: int) -> int:
    """
    Get the distance between country A and B based on the simple gravity trade model 

    """

    GDP_A = gdp(countryA, year, quarter)
    GDP_B = gdp(countryB, year, quarter)
    dist = geo_distance(countryA, countryB)
    
    return GDP_A * GDP_B / dist

def proposalDistance(countryA: str, countryB: str, year: int, quarter: int) -> int:
     theta = 0
     gamma = 0
     dist = geo_distance(countryA, countryB) + theta
     return dist/(math.log(gdp(countryA, year, quarter))- math.log(gdp(countryB, year, quarter)))^2
