
import pandas as pd
import matplotlib.pyplot as plt

# Load the extracted features
df = pd.read_csv("data/processed/ventilator_features.csv")

print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of recordings:")
print(df["recording_id"].nunique())

print("\nNumber of breaths:")
print(len(df))

print("\nSummary statistics:")
print(df.describe())

print("\nSample count per breath:")
print(df["sample_count"].describe())

# Check for very short breaths
short_breaths = df[df["sample_count"] < 10]

print("\nBreaths with fewer than 10 samples:")
print(len(short_breaths))

# Plot distribution of sample counts
plt.figure(figsize=(10, 5))

plt.hist(df["sample_count"], bins=50)

plt.xlabel("Samples per Breath")
plt.ylabel("Number of Breaths")
plt.title("Distribution of Ventilator Breath Lengths")
plt.grid(True)

plt.tight_layout()
plt.show()
