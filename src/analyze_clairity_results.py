
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Project folders
project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "processed"
images_folder = project_folder / "images"

images_folder.mkdir(exist_ok=True)

# Load results and Version 2 dataset
results = pd.read_csv(
    data_folder / "model_comparison_v2.csv"
)

df = pd.read_csv(
    data_folder / "clairity_hybrid_v2.csv"
)

# ---------------------------------------
# GRAPH 1: Model accuracy comparison
# ---------------------------------------

plt.figure(figsize=(9, 5))

names = results["model"]
accuracy = results["accuracy"] * 100

plt.bar(names, accuracy)

plt.title("CLAIRITY Version 2 - Model Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for i, value in enumerate(accuracy):
    plt.text(i, value + 2, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.savefig(images_folder / "model_comparison_v2.png", dpi=300)
plt.show()

# ---------------------------------------
# GRAPH 2: Hybrid confusion matrix
# From the previous training results
# ---------------------------------------

matrix = np.array([
    [2128, 232, 16, 0],
    [525, 750, 100, 0],
    [55, 199, 391, 0],
    [0, 0, 0, 381]
])

labels = [
    "Normal",
    "At Risk",
    "Critical",
    "Sensor Failure"
]

plt.figure(figsize=(8, 6))
plt.imshow(matrix, cmap="Blues")
plt.colorbar(label="Number of predictions")

plt.xticks(range(4), labels, rotation=30)
plt.yticks(range(4), labels)

plt.xlabel("Predicted condition")
plt.ylabel("Actual condition")
plt.title("CLAIRITY Hybrid Model - Confusion Matrix")

for i in range(4):
    for j in range(4):
        plt.text(
            j, i, str(matrix[i, j]),
            ha="center", va="center"
        )

plt.tight_layout()
plt.savefig(images_folder / "hybrid_confusion_matrix_v2.png", dpi=300)
plt.show()

# ---------------------------------------
# GRAPH 3: Class recall
# ---------------------------------------

recall = np.diag(matrix) / matrix.sum(axis=1)

plt.figure(figsize=(9, 5))
plt.bar(labels, recall * 100)

plt.title("CLAIRITY Hybrid Model - Recall by Condition")
plt.ylabel("Recall (%)")
plt.ylim(0, 110)

for i, value in enumerate(recall):
    plt.text(i, value * 100 + 2, f"{value * 100:.1f}%",
             ha="center")

plt.tight_layout()
plt.savefig(images_folder / "hybrid_recall_v2.png", dpi=300)
plt.show()

# ---------------------------------------
# GRAPH 4: Simulated sensor distributions
# ---------------------------------------

plt.figure(figsize=(9, 5))

for condition in ["normal", "at_risk", "critical"]:
    values = df.loc[
        df["condition"] == condition,
        "capacitance_below"
    ].dropna()

    plt.hist(
        values,
        bins=40,
        alpha=0.45,
        density=True,
        label=condition
    )

plt.title("Simulated Capacitance Below Cuff")
plt.xlabel("Simulated sensor value (arbitrary units)")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.savefig(images_folder / "sensor_distribution_v2.png", dpi=300)
plt.show()

print("\nAll four graphs created successfully!")
print("Saved in:", images_folder)
