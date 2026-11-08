
from pathlib import Path

import pandas as pd
import plotly.express as px

# 1. Set up folders
project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "processed"
plots_folder = project_folder / "plots"

plots_folder.mkdir(exist_ok=True)

# 2. Load the saved hybrid predictions
df = pd.read_csv(
    data_folder / "clairity_hybrid_predictions_v2.csv"
)

print("CLAIRITY ERROR ANALYSIS")
print("----------------------")
print("Total test records:", len(df))

# 3. Find correct and incorrect predictions
df["correct_prediction"] = (
    df["condition"] == df["predicted_condition"]
)

accuracy = df["correct_prediction"].mean() * 100
errors = df[~df["correct_prediction"]].copy()

print(f"Accuracy: {accuracy:.2f}%")
print("Incorrect predictions:", len(errors))

# 4. Count each type of mistake
error_counts = (
    errors.groupby(["condition", "predicted_condition"])
    .size()
    .reset_index(name="count")
)

print("\nTypes of prediction errors:")
print(error_counts.to_string(index=False))

# 5. Interactive confusion matrix
labels = ["normal", "at_risk", "critical", "sensor_failure"]

matrix = pd.crosstab(
    df["condition"],
    df["predicted_condition"]
).reindex(index=labels, columns=labels, fill_value=0)

fig = px.imshow(
    matrix,
    text_auto=True,
    color_continuous_scale="Blues",
    labels={
        "x": "Predicted condition",
        "y": "Actual condition",
        "color": "Number of cases"
    },
    title="CLAIRITY V2 - Interactive Confusion Matrix"
)

fig.write_html(
    plots_folder / "interactive_confusion_matrix.html",
    include_plotlyjs=True
)

# 6. Interactive prediction errors
fig = px.bar(
    error_counts,
    x="condition",
    y="count",
    color="predicted_condition",
    barmode="group",
    title="CLAIRITY V2 - Prediction Errors",
    labels={
        "condition": "Actual condition",
        "count": "Number of mistakes",
        "predicted_condition": "Predicted as"
    }
)

fig.write_html(
    plots_folder / "interactive_prediction_errors.html",
    include_plotlyjs=True
)

# 7. Interactive sensor distributions
for sensor in [
    "capacitance_above",
    "capacitance_below",
    "ph_above",
    "ph_below"
]:
    sensor_df = df.dropna(subset=[sensor]).copy()

    fig = px.histogram(
        sensor_df,
        x=sensor,
        color="condition",
        histnorm="probability density",
        barmode="overlay",
        opacity=0.55,
        nbins=50,
        title=f"CLAIRITY V2 - {sensor}",
        labels={
            "condition": "Actual simulated condition"
        }
    )

    fig.write_html(
        plots_folder / f"interactive_{sensor}.html",
        include_plotlyjs=True
    )

# 8. Save error records
errors.to_csv(
    data_folder / "clairity_errors_v2.csv",
    index=False
)

print("\nAnalysis complete!")
print("Interactive graphs saved in:", plots_folder)
print("Error records saved in:", data_folder)
