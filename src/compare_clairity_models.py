
from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------
# 1. Project folders
# ---------------------------------------

project_folder = Path(__file__).resolve().parent.parent

data_folder = project_folder / "data" / "processed"

data_path = data_folder / "clairity_hybrid_v2.csv"

output_folder = data_folder
output_folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# 2. Load dataset
# ---------------------------------------

df = pd.read_csv(data_path)

print("CLAIRITY VERSION 2 MODEL COMPARISON")
print("Dataset shape:", df.shape)

# ---------------------------------------
# 3. Define features
# ---------------------------------------

ventilator_features = [
    "sample_count",
    "signal_1_mean",
    "signal_1_min",
    "signal_1_max",
    "signal_1_std",
    "signal_1_range",
    "signal_2_mean",
    "signal_2_min",
    "signal_2_max",
    "signal_2_std",
    "signal_2_range",
]

sensor_features = [
    "capacitance_above",
    "capacitance_below",
    "ph_above",
    "ph_below",
]

hybrid_features = ventilator_features + sensor_features

feature_groups = {
    "ventilator_only": ventilator_features,
    "sensors_only": sensor_features,
    "hybrid": hybrid_features,
}

# ---------------------------------------
# 4. Prepare labels and recording groups
# ---------------------------------------

y = df["condition"]

groups = df["recording_id"]

# Split by recording, not individual breath
# This prevents the same recording from
# appearing in both training and testing.

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42,
)

train_index, test_index = next(
    splitter.split(df, y, groups=groups)
)

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

print("\nTraining records:", len(train_index))
print("Testing records:", len(test_index))

# ---------------------------------------
# 5. Train and compare models
# ---------------------------------------

results = []

labels = [
    "at_risk",
    "critical",
    "normal",
    "sensor_failure",
]

for model_name, features in feature_groups.items():

    print("\n" + "=" * 45)
    print("MODEL:", model_name)
    print("=" * 45)

    X = df[features]

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # -----------------------------------
    # Save hybrid predictions
    # -----------------------------------

    if model_name == "hybrid":

        error_df = df.iloc[test_index].copy()

        error_df["predicted_condition"] = predictions

        error_df["correct_prediction"] = (
            error_df["condition"]
            == error_df["predicted_condition"]
        )

        error_path = (
            output_folder
            / "clairity_hybrid_predictions_v2.csv"
        )

        error_df.to_csv(error_path, index=False)

        print("\nHybrid predictions saved to:", error_path)

        # Save the trained hybrid model
        joblib.dump(
            {
                "model": model,
                "features": features,
                "labels": labels,
            },
            project_folder / "data" / "clairity_model_v2.pkl",
        )

    # -----------------------------------
    # Evaluate EVERY model
    # This must stay outside the if block
    # -----------------------------------

    accuracy = accuracy_score(y_test, predictions)

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        )
    )

    results.append(
        {
            "model": model_name,
            "accuracy": accuracy,
        }
    )

# ---------------------------------------
# 6. Save comparison
# ---------------------------------------

results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(results_df)

comparison_path = (
    output_folder / "model_comparison_v2.csv"
)

results_df.to_csv(comparison_path, index=False)

print("\nSaved to:", comparison_path)

print("\nAll three models evaluated successfully!")
