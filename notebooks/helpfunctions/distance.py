import pandas as pd
import pycountry as pyc


def distance(countryA: str, countryB: str) -> int:
    """
    Get the distance between country A and B 

    """

    filepath="./../../assets/dist_cepii.xls"
    df_distance = pd.read_excel(filepath)
    df_distance.head
    countryA = pyc.countries.get(name= countryA)
    countryB = pyc.countries.get(name = countryB)
    row = df_distance[(df_distance['iso_o'] == countryA.alpha_3) & (df_distance['iso_d'] == countryB.alpha_3)]
    dist = row['dist'].item()
    return dist
     
