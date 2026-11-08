import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# read the data
df = pd.read_csv("data/synthetic_sensor_data.csv")

# choose the sensor measurements
features = [
    "cuff_pressure",
    "fluid_signal",
    "volume_difference_pct",
    "ph_above_cuff"
]

# fill missing values so the model can use the data
df[features] = df[features].fillna(-1)

X = df[features]
y = df["scenario"]

# split by simulation run, not by individual reading
split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

train_index, test_index = next(
    split.split(X, y, groups=df["run_id"])
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

# create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
joblib.dump(model, "data/auradx_model.pkl")
print("Model saved!")

# predict using the test data
predictions = model.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("AuraDx Machine Learning Results")
print("------------------------------")
print("Training readings:", len(X_train))
print("Testing readings:", len(X_test))
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=model.classes_))
print("Class order:", list(model.classes_))
