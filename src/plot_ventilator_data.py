
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned sample
df = pd.read_csv("data/processed/ventilator_sample.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Choose the first recording
recording_id = df["recording_id"].iloc[0]
sample = df[df["recording_id"] == recording_id]

# Plot the first 500 measurements
sample = sample.head(500)

plt.figure(figsize=(12, 5))

plt.plot(
    sample["sample_number"],
    sample["signal_1"],
    label="Signal 1"
)

plt.plot(
    sample["sample_number"],
    sample["signal_2"],
    label="Signal 2"
)

plt.xlabel("Sample Number")
plt.ylabel("Recorded Value")
plt.title("Real Ventilator Waveform Sample")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()