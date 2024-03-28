import pandas as pd
import numpy as np
from scipy.integrate import quad
from seminartools.models.base_model import BaseModel
from scipy.stats import ecdf

def getMSPE(data : pd.Dataframe):
    """
        args: dataframe containing both the actual values ("actual") and the predicted values ("predicted") of a variable
        returns MSPE of a model, if the actual and prediction columns are named "pred" and "actual" respectively
    """
    E = data.pred - data.actual
    SPE = E**2
    SSPE= sum(SPE)
    MSPE = SSPE/len(data)
    return MSPE

#in progress, depends on type of density forecasts we will produce
def getUpsideEntropy(modelDensity : BaseModel, actual : pd.DataFrame):
    unconditionalCDF = ecdf(actual)
    unconditionalDensity = 
    diff = (np.log(unconditionalDensity) - np.log(modelDensity))
    internalFunction = diff*modelDensity

    median = modelDensity.median
    integral = quad(internalFunction,-np.inf, median)


