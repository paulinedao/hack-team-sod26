
%%writefile /content/drive/MyDrive/more-than-a-hack-2027/more-than-a-hack-2027/starter/my_model.py

import os
import sys

ROOT = "/content/drive/MyDrive/more-than-a-hack-2027/more-than-a-hack-2027"

sys.path.insert(0, ROOT)

from starter.baseline import build, smooth, write_submission

# Data directory
DATA_DIR = os.environ["MTAH_DATA_DIR"]

# Generate raw tracking predictions
codes = build(data_dir=DATA_DIR)

# Experiment with different smoothing windows
WINDOWS = [25, 50, 75, 100, 125]

OUTPUT_DIR = os.path.join(ROOT, "evaluation", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for window in WINDOWS:

    labels = smooth(codes, window)

    filename = f"my_model_smooth{window}.csv"

    output_path = os.path.join(OUTPUT_DIR, filename)

    write_submission(output_path, labels)

    print(
        f"Generated {filename}: "
        f"H={(labels == 'H').sum()}, "
        f"A={(labels == 'A').sum()}, "
        f"D={(labels == 'D').sum()}"
    )

print("All predictions generated!")
