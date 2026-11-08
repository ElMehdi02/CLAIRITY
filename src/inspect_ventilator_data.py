
from pathlib import Path
import zipfile

# Find the ZIP file
project_folder = Path(__file__).resolve().parent.parent

zip_path = (
    project_folder
    / "data"
    / "raw"
    / "ventilator_waveform_data.zip"
)

# Open the ZIP file
with zipfile.ZipFile(zip_path, "r") as zip_file:

    csv_files = sorted([
        name for name in zip_file.namelist()
        if name.lower().endswith(".csv")
    ])

    print("Total recordings:", len(csv_files))

    # Inspect the first recording
    file_name = csv_files[0]

    print("\nRecording:", file_name)

    with zip_file.open(file_name) as file:
        content = file.read(20000).decode(
            "utf-8", errors="replace"
        )

    content = content.replace("\x00", "")

    print("\nFirst 35 lines:")
    print("-" * 40)

    for line in content.splitlines()[:35]:
        print(line)

    print("-" * 40)

    # Look for recording markers
    print("\nRecording markers:")

    markers = [
        line for line in content.splitlines()
        if line.startswith(("BS", "BE"))
    ]

    for marker in markers[:10]:
        print(marker)
