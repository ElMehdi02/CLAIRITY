
from pathlib import Path
import numpy as np
import pandas as pd

# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

input_path = (
    project_folder / "data" / "processed" / "ventilator_features.csv"
)

output_path = (
    project_folder / "data" / "processed" / "clairity_hybrid_v2.csv"
)

# Load real ventilator features
df = pd.read_csv(input_path)

# Select 20,000 breaths
df = df.sample(n=20000, random_state=42).copy()

# Keep breaths in their original order within recordings
df = df.sort_index().reset_index(drop=True)

rng = np.random.default_rng(42)
n = len(df)

# Simulated physical conditions:
# 0 = no fluid
# 1 = fluid above cuff
# 2 = possible fluid movement below cuff

fluid_state = rng.choice(
    [0, 1, 2],
    size=n,
    p=[0.55, 0.30, 0.15]
)

# Simulated sensor failure
sensor_failure = rng.random(n) < 0.08

# Generate fluid levels with overlap between states
fluid_above = np.zeros(n)
fluid_below = np.zeros(n)

for state in [0, 1, 2]:

    mask = fluid_state == state
    count = mask.sum()

    if state == 0:
        fluid_above[mask] = rng.normal(1.5, 1.0, count)
        fluid_below[mask] = rng.normal(0.8, 0.6, count)

    elif state == 1:
        fluid_above[mask] = rng.normal(4.0, 1.5, count)
        fluid_below[mask] = rng.normal(1.2, 0.8, count)

    else:
        fluid_above[mask] = rng.normal(5.5, 1.8, count)
        fluid_below[mask] = rng.normal(3.8, 1.5, count)

fluid_above = np.clip(fluid_above, 0, None)
fluid_below = np.clip(fluid_below, 0, None)

# Add measurement noise
df["capacitance_above"] = (
    fluid_above + rng.normal(0, 1.0, n)
)

df["capacitance_below"] = (
    fluid_below + rng.normal(0, 1.0, n)
)

# Simulated pH readings
# These are illustrative values, not calibrated HPTS data.
df["ph_above"] = (
    7.2 - 0.15 * fluid_above
    + rng.normal(0, 0.5, n)
)

df["ph_below"] = (
    7.2 - 0.15 * fluid_below
    + rng.normal(0, 0.5, n)
)

# Add gradual sensor drift within each recording
for recording_id, group in df.groupby("recording_id", sort=False):

    positions = group.index.to_numpy()
    drift = np.linspace(
        0, rng.normal(0, 0.8), len(group)
    )

    df.loc[positions, "capacitance_above"] += drift
    df.loc[positions, "capacitance_below"] += drift

# Keep pH within its possible range
df["ph_above"] = df["ph_above"].clip(0, 14)
df["ph_below"] = df["ph_below"].clip(0, 14)

# Simulate missing measurements
df.loc[sensor_failure, "capacitance_below"] = np.nan
df.loc[sensor_failure, "ph_below"] = np.nan

# Assign labels from the simulated underlying state
conditions = np.array(["normal", "at_risk", "critical"])

df["condition"] = conditions[fluid_state]

# Sensor failure takes priority in this first experiment
df.loc[sensor_failure, "condition"] = "sensor_failure"

# Save the new dataset
df.to_csv(output_path, index=False)

print("\nCLAIRITY VERSION 2 DATASET CREATED")
print("----------------------------------")

print("Dataset shape:", df.shape)

print("\nCondition counts:")
print(df["condition"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSaved to:")
print(output_path)
