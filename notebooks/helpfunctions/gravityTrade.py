import gdp, distance

def gravityTrade(countryA: str, countryB: str, year: int, quarter: int) -> int:
    """
    Get the distance between country A and B based on the simple gravity trade model 

    """

    GDP_A = gdp.gdp(countryA, year, quarter)
    GDP_B = gdp.gdp(countryB, year, quarter)
    dist = distance.distance(countryA, countryB)
    
    return GDP_A * GDP_B / dist

    