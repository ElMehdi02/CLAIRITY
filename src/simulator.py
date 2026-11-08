
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

records = []

scenarios = [
    "normal",
    "fluid_buildup",
    "possible_cuff_leak",
    "sensor_failure"
]

for run_id in range(40):
    scenario = scenarios[run_id % len(scenarios)]

    for second in range(120):
        # Placeholder values, NOT clinical thresholds
        cuff_pressure = 25 + rng.normal(0, 1)
        fluid_signal = 0.1 + rng.normal(0, 0.02)
        volume_difference = 5 + rng.normal(0, 2)
        ph = 7.0 + rng.normal(0, 0.1)

        if scenario == "fluid_buildup":
            fluid_signal += 0.3

        elif scenario == "possible_cuff_leak":
            cuff_pressure -= 6
            volume_difference += 15
            fluid_signal += 0.15

        elif scenario == "sensor_failure":
            cuff_pressure = np.nan

        records.append({
            "run_id": run_id,
            "time_seconds": second,
            "scenario": scenario,
            "cuff_pressure": cuff_pressure,
            "fluid_signal": fluid_signal,
            "volume_difference_pct": volume_difference,
            "ph_above_cuff": ph
        })

df = pd.DataFrame(records)

output = Path(__file__).resolve().parent.parent
output = output / "data" / "synthetic_sensor_data.csv"

output.parent.mkdir(exist_ok=True)
df.to_csv(output, index=False)

print(f"Generated {len(df)} sensor readings")
print(df["scenario"].value_counts())
print(f"Saved to: {output}")
