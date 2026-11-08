
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project folders
project_folder = Path(__file__).resolve().parent.parent

data_path = (
    project_folder / "data" / "processed" / "clairity_hybrid_data.csv"
)

model_path = (
    project_folder / "data" / "clairity_model.pkl"
)

# Load the dataset
df = pd.read_csv(data_path)

# Real ventilator features
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
    "signal_2_range"
]

# Simulated CLAIRITY sensor features
sensor_features = [
    "capacitance_above",
    "capacitance_below",
    "ph_above",
    "ph_below"
]

all_features = ventilator_features + sensor_features

# Target labels
y = df["condition"]

# Keep recordings separate between training and testing
split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_index, test_index = next(
    split.split(df, y, groups=df["recording_id"])
)

X_train = df.iloc[train_index][all_features]
X_test = df.iloc[test_index][all_features]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Random Forest can handle missing values in supported
# scikit-learn versions, but we fill them explicitly
# for compatibility with different versions.
#
# Missing values are also useful information because
# they represent simulated sensor failures.

for feature in sensor_features:
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[feature + "_missing"] = (
        X_train[feature].isna().astype(int)
    )

    X_test[feature + "_missing"] = (
        X_test[feature].isna().astype(int)
    )

    median_value = X_train[feature].median()

    X_train[feature] = X_train[feature].fillna(median_value)
    X_test[feature] = X_test[feature].fillna(median_value)

# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nCLAIRITY MODEL RESULTS")
print("----------------------")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model and feature names
joblib.dump(
    {
        "model": model,
        "features": X_train.columns.tolist(),
        "sensor_medians": {
            feature: df.iloc[train_index][feature].median()
            for feature in sensor_features
        }
    },
    model_path
)

print("\nModel saved to:", model_path)
