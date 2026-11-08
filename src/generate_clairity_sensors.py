
from pathlib import Path
import numpy as np
import pandas as pd

# Find our project folder
project_folder = Path(__file__).resolve().parent.parent

input_path = (
    project_folder / "data" / "processed" / "ventilator_features.csv"
)

output_path = (
    project_folder / "data" / "processed" / "clairity_hybrid_data.csv"
)

# Load the real ventilator features
df = pd.read_csv(input_path)

# Select 20,000 breaths
df = df.sample(n=20000, random_state=42).copy()

# Random number generator
rng = np.random.default_rng(42)

# Create simulated conditions
conditions = [
    "normal",
    "at_risk",
    "critical",
    "sensor_failure"
]

df["condition"] = rng.choice(
    conditions,
    size=len(df),
    p=[0.40, 0.30, 0.20, 0.10]
)

# Generate simulated sensor readings
df["capacitance_above"] = 0.0
df["capacitance_below"] = 0.0
df["ph_above"] = 0.0
df["ph_below"] = 0.0

for condition in conditions:

    mask = df["condition"] == condition
    n = mask.sum()

    if condition == "normal":

        df.loc[mask, "capacitance_above"] = rng.normal(
            2.0, 0.5, n
        )

        df.loc[mask, "capacitance_below"] = rng.normal(
            1.0, 0.3, n
        )

        df.loc[mask, "ph_above"] = rng.normal(
            7.0, 0.3, n
        )

        df.loc[mask, "ph_below"] = rng.normal(
            7.1, 0.3, n
        )

    elif condition == "at_risk":

        df.loc[mask, "capacitance_above"] = rng.normal(
            5.0, 1.0, n
        )

        df.loc[mask, "capacitance_below"] = rng.normal(
            1.5, 0.5, n
        )

        df.loc[mask, "ph_above"] = rng.normal(
            6.5, 0.4, n
        )

        df.loc[mask, "ph_below"] = rng.normal(
            7.0, 0.3, n
        )

    elif condition == "critical":

        df.loc[mask, "capacitance_above"] = rng.normal(
            7.0, 1.2, n
        )

        df.loc[mask, "capacitance_below"] = rng.normal(
            5.0, 1.0, n
        )

        df.loc[mask, "ph_above"] = rng.normal(
            5.8, 0.5, n
        )

        df.loc[mask, "ph_below"] = rng.normal(
            6.0, 0.5, n
        )

    elif condition == "sensor_failure":

        df.loc[mask, "capacitance_above"] = rng.normal(
            2.0, 0.5, n
        )

        df.loc[mask, "capacitance_below"] = np.nan

        df.loc[mask, "ph_above"] = rng.normal(
            7.0, 0.3, n
        )

        df.loc[mask, "ph_below"] = np.nan

# Keep pH within the possible pH scale
df["ph_above"] = df["ph_above"].clip(0, 14)
df["ph_below"] = df["ph_below"].clip(0, 14)

# Save our new hybrid dataset
df.to_csv(output_path, index=False)

print("CLAIRITY hybrid dataset created!")

print("\nDataset shape:")
print(df.shape)

print("\nCondition counts:")
print(df["condition"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSaved to:")
print(output_path)
