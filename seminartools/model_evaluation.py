import pandas as pd
import numpy as np
from scipy.integrate import quad
from seminartools.models.base_model import BaseModel
from sklearn.metrics import mean_squared_error

def get_mspe(data : pd.DataFrame):
#     """
#         args: dataframe containing both the actual values ("actual") and the predicted values ("predicted") of a variable
#         returns MSPE of a model, if the actual and prediction columns are named "pred" and "actual" respectively
#     """
#     E = data.pred - data.actual
#     SPE = E**2
#     SSPE= sum(SPE)
#     MSPE = SSPE/len(data)
#     return MSPE
    return mean_squared_error(data.actual, data.pred)

# This method above is exactly the same as sklearn mean_squared_error

#in progress, depends on type of density forecasts we will produce
def get_upside_entropy(modelDensity : BaseModel, unconditionalDensity : BaseModel):
    raise NotImplementedError()

    diff = (np.log(unconditionalDensity) - np.log(modelDensity))
    internalFunction = diff*modelDensity
    
    

    median = modelDensity.median
    integral = quad(internalFunction,-np.inf, median)

def get_95_interval(pdf_values,inflation_data):
    sum_left= 0
    sumRight = 0
    for i in range(0,len(pdf_values)):
        sum_left +=pdf_values[i]
        if sum_left >= 0.025:
            left_index = i
            break
        
    for i in range (0,len(pdf_values)):
        sumRight += pdf_values[-i]
        if sumRight >=0.025:
            right_index = i
            break

    interval = [inflation_data[left_index], inflation_data[-right_index]]
    return interval