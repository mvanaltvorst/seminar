import os

# 12 cores
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=12"

import jax

jax.config.update("jax_platform_name", "cpu")

import pandas as pd
from seminartools.data import read_inflation
from seminartools.models.mucsvss_model import MUCSVSSModel
import argparse

parser = argparse.ArgumentParser(description="Fit MUCSVSS model to inflation data")
parser.add_argument(
    "--num_particles",
    type=int,
    default=100008,
    help="Number of particles for the model",
)
parser.add_argument(
    "--stochastic_seasonality",
    type=bool,
    default=True,
    help="Whether to include stochastic seasonality",
)
parser.add_argument(
    "--countries",
    type=str,
    default="all",
    help="Countries to fit the model to, separated by commas",
)
args = parser.parse_args()

filepath = f"../../models/mucsvss_model_{args.num_particles}_{'stochastic' if args.stochastic_seasonality else 'deterministic'}_{'all' if args.countries == 'all' else args.countries}.pkl"

if os.path.exists(filepath):
    print("Model already exists, skipping...")
    exit()


df_inflation = read_inflation(mergeable_format=True).reset_index()
if args.countries != "all":
    df_inflation = df_inflation[df_inflation["country"].isin(args.countries.split(","))]


model = MUCSVSSModel(
    num_particles=args.num_particles, stochastic_seasonality=args.stochastic_seasonality
)
model.full_fit(df_inflation)
model.save_to_disk(filepath)


print("Done!")
