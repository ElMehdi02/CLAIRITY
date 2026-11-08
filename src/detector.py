import pandas as pd

# read the sensor data
df = pd.read_csv("data/synthetic_sensor_data.csv")

# function to detect problems
def detect_problem(row):

    # check if any sensor has missing data
    if pd.isna(row["cuff_pressure"]) or pd.isna(row["fluid_signal"]) or pd.isna(row["volume_difference_pct"]) or pd.isna(row["ph_above_cuff"]):
        return "sensor_failure"

    # check for a possible cuff leak
    if row["cuff_pressure"] < 21 and row["volume_difference_pct"] > 12:
        return "possible_cuff_leak"

    # check for fluid buildup
    if row["fluid_signal"] > 0.25:
        return "fluid_buildup"

    return "normal"


# apply the function to all rows
df["prediction"] = df.apply(detect_problem, axis=1)

# calculate accuracy
accuracy = (df["scenario"] == df["prediction"]).mean()

print("AuraDx Detection Results")
print("-----------------------")
print("Total readings:", len(df))
print("Accuracy:", round(accuracy * 100, 2), "%")

# compare actual and predicted results
print("\nActual vs Predicted:")
print(pd.crosstab(df["scenario"], df["prediction"]))

# save the results
df.to_csv("data/detection_results.csv", index=False)

print("\nResults saved!")