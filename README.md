# Seminar
[Final report](final_report.pdf)

By Maurits van Altvorst, Philipp Hoecker, Matthias Hofstede and Stefan van Diepen.

This project is structured into different parts:

1. Python package: the `seminartools` package provides us with models and utilities created during this seminar.
2. Notebooks: this is the glue that merges all models and utilities together. 

## Installation of `seminartools`

```bash
cd seminartools/
pip install -e .
```

Changes to the package will be directly reflected in your current venv.

# Structure of `seminartools` package
### `seminartools.models`
Contains several models:
- ARMAX
- covariance BHM
- distance BHM
- 4 quarter average model (AO)
- MUCSVSS model
    - This model is expensive to fit (`full_fit(.)` takes approx. 1-2 hours and up to 80GB of RAM). We built in functionality to save this model to disk using pickle files and load it back later. These fitted models are found in `models/` and are generated using the `notebooks/models/mucsvssmodel_long_run.py` CLI script (use with `-h` argument to see all command line arguments). 
- UCSVSS model
    - pass argument `stochastic_seasonality=False` to obtain UCSV model
- PCAVAR model
- VAR model
    - not directly used on panel inflation data, but is used implicitly by the PCAVAR model

Each of these models inherits from the `BaseModel` class and implements the following methods:
- `fit(df_inflation)`: fits the model given all data available in `df_inflation`. 
- `predict(df_inflation)`: given `df_inflation`, predict cross-sectional inflation at time `max(t) + 1` where `max(t)` is the maximum timestamp occurring in `df_inflation`. 

If `Model.FULL_FIT == True`, the model is path-dependent and cannot be used to make far out-of-sample predictions (e.g. the models that depend on particle filters). We fit these using the `full_fit(.)` method on the entire dataset (train + test) and depend on internal logic within the model to ensure there are no lookahead errors. 

### `seminartools.models.utils`
In `models/utils.py`, we have methods to calculate e.g. h-period ahead forecasts given a model, perform expanding window retraining, and calculate MSE/MAE/Mincer Zarnowitz statistics. 

### `seminartools.data`
Methods to read data from the many data sources we employ in a standardized format. Also exports `read_merged` which provides the intersection of all available data across data sources.

### `seminartools.matern_kernel`
A Matern kernel that's used by the distance model. It takes in a distance function and returns a covariance matrix.

### `seminartools.model_evaluation` and `seminartools.modelEvaluationTests`
Miscellaneous experimental helper functions that were not used in the end.

### `seminartools.time_series_split`
Contains an expanding window time series splitting method that is used to retrain every 4 years.

### `seminartools.utils`
Geospatial distance, country -> continent function, and other geospatial helper functions.

# Structure of notebooks
### `notebooks/data`
Helper notebooks used to experiment with the data and how to merge these together.

### `notebooks/methods`
Helper notebooks used to experiment with the particle filters and BHMs. These were only experiments regarding the methodology itself; the true fully fledged models with corresponding experiemts can be found within `seminartools.models` and `notebooks/models`.

### `notebooks/models`
Contains a notebook corresponding to each model to experiment with the efficacy of said model. 

### `notebooks/model_eval`
The important results lie here. `model_eval.ipynb` contains the code that results in the main 1 quarter ahead inflation prediction results. 


## Useful references
- https://florianwilhelm.info/2020/10/bayesian_hierarchical_modelling_at_scale/
- https://www.pymc.io/welcome.html
