import os
import csv

# Path to your RAVDESS dataset folder (update if needed)
base_dir = r"C:\Users\mlaxm\OneDrive\Desktop\vesper\Vesper\wav\archive (2)"

# Output CSV file
output_csv = "metadata.csv"

# RAVDESS emotion ID to integer label
emotion_map = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fearful
    "07": 6,  # disgust
    "08": 7   # surprised
}

rows = [("name", "label")]

# Walk through the dataset folder and find all .wav files
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split("-")
            if len(parts) >= 3:
                emotion_id = parts[2]
                label = emotion_map.get(emotion_id)
                if label is not None:
                    rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                    rel_path = rel_path.replace("\\", "/")  # Normalize for Windows
                    name = rel_path.replace(".wav", "")
                    rows.append((name, label))

# Write to CSV
with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"✅ metadata.csv created with {len(rows)-1} entries at: {os.path.abspath(output_csv)}")
