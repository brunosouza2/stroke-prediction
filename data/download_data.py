"""
Download the Stroke Prediction Dataset from Kaggle.

Usage:
    python data/download_data.py

Requires kagglehub (included in requirements.txt).
The CSV will be saved to data/healthcare-dataset-stroke-data.csv.
"""

import os
import shutil
import kagglehub

DATASET_ID = "fedesoriano/stroke-prediction-dataset"
CSV_NAME = "healthcare-dataset-stroke-data.csv"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def download():
    print(f"Downloading dataset '{DATASET_ID}' from Kaggle...")
    path = kagglehub.dataset_download(DATASET_ID)
    src = os.path.join(path, CSV_NAME)
    dst = os.path.join(OUTPUT_DIR, CSV_NAME)
    shutil.copy(src, dst)
    print(f"Dataset saved to: {dst}")
    return dst


if __name__ == "__main__":
    download()
