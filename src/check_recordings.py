
from pathlib import Path
import zipfile
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent

zip_path = (
    project_folder
    / "data"
    / "raw"
    / "ventilator_waveform_data.zip"
)

df = pd.read_csv(
    project_folder / "data" / "processed" / "ventilator_features.csv"
)

with zipfile.ZipFile(zip_path, "r") as zip_file:
    csv_files = sorted(
        name for name in zip_file.namelist()
        if name.lower().endswith(".csv")
    )

print("Original CSV files:", len(csv_files))
print("Unique recording IDs in processed data:", df["recording_id"].nunique())

# Find duplicate filename stems
names = [Path(name).stem for name in csv_files]

name_counts = pd.Series(names).value_counts()

duplicates = name_counts[name_counts > 1]

print("\nDuplicate recording names:")
print(duplicates)

# Find files with no extracted breaths
processed_names = set(df["recording_id"])

missing = [
    name for name in names
    if name not in processed_names
]

print("\nFiles without extracted breaths:")
print(missing)

# Show how many breaths each recording contains
print("\nBreaths per recording:")
print(df.groupby("recording_id").size().describe())
