import os
# 12 cores
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=12'

import jax
jax.config.update('jax_platform_name', 'cpu')
import pandas as pd
from seminartools.data import read_inflation
from seminartools.models.mucsvss_model import MUCSVSSModel

df_inflation = read_inflation(mergeable_format=True).reset_index()

model = MUCSVSSModel(
    num_particles=100008, stochastic_seasonality=True
)
model.full_fit(df_inflation)
model.save_to_disk()

print("Done!")