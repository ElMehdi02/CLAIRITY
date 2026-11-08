
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# Project folders
project_folder = Path(__file__).resolve().parent.parent

zip_path = (
    project_folder
    / "data"
    / "raw"
    / "ventilator_waveform_data.zip"
)

output_folder = project_folder / "data" / "processed"
output_folder.mkdir(parents=True, exist_ok=True)

output_path = output_folder / "ventilator_features.csv"

# Store one row per breath
all_breaths = []

with zipfile.ZipFile(zip_path, "r") as zip_file:

    csv_files = sorted([
        name for name in zip_file.namelist()
        if name.lower().endswith(".csv")
    ])

    print("Total recordings:", len(csv_files))

    for file_number, file_name in enumerate(csv_files, start=1):

        print(f"Processing {file_number}/{len(csv_files)}")

        recording_id = Path(file_name).stem
        breath_id = None

        signal_1 = []
        signal_2 = []

        # Process one breath at a time
        def save_breath():

            if len(signal_1) < 2:
                return

            a = np.array(signal_1)
            b = np.array(signal_2)

            all_breaths.append({
                "recording_id": recording_id,
                "breath_id": breath_id,
                "sample_count": len(a),

                "signal_1_mean": a.mean(),
                "signal_1_min": a.min(),
                "signal_1_max": a.max(),
                "signal_1_std": a.std(),
                "signal_1_range": np.ptp(a),

                "signal_2_mean": b.mean(),
                "signal_2_min": b.min(),
                "signal_2_max": b.max(),
                "signal_2_std": b.std(),
                "signal_2_range": np.ptp(b)
            })

        # Read the recording line by line
        with zip_file.open(file_name) as file:

            for raw_line in file:

                line = raw_line.decode(
                    "utf-8", errors="replace"
                ).replace("\x00", "").strip()

                if line.startswith("BS"):

                    save_breath()

                    breath_id = line
                    signal_1 = []
                    signal_2 = []
                    continue

                if line.startswith("BE"):

                    save_breath()

                    breath_id = None
                    signal_1 = []
                    signal_2 = []
                    continue

                if breath_id is None:
                    continue

                values = line.split(",")

                if len(values) != 2:
                    continue

                try:
                    value_1 = float(values[0])
                    value_2 = float(values[1])

                    if np.isfinite(value_1) and np.isfinite(value_2):
                        signal_1.append(value_1)
                        signal_2.append(value_2)

                except ValueError:
                    continue

        # Save the final breath if the file ended
        save_breath()

# Create a smaller dataset
df = pd.DataFrame(all_breaths)

df.to_csv(output_path, index=False)

print("\nData preparation completed!")
print("Total recordings:", len(csv_files))
print("Total breaths:", len(df))
print("Dataset shape:", df.shape)
print("Saved to:", output_path)

print("\nFirst five rows:")
print(df.head())