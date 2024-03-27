import pandas as pd
import numpy as np
import scipy.integrate as integrate
from seminartools.models.base_model import BaseModel

def getMSPE(data : pd.Dataframe):
    """
        returns MSPE of a model, if the actual and prediction columns are named "pred" and "actual" respectively
    """
    E = data.pred - data.actual
    SPE = E**2
    SSPE= sum(SPE)
    MSPE = SSPE/len(data)
    return MSPE

#in progress, depends on type of density forecasts we will produce
def getUpsideEntropy(modelDensity : BaseModel, unconditionalDensity : BaseModel):
    diff = (np.log(unconditionalDensity) - np.log(modelDensity))
    internalFunction = diff*modelDensity
