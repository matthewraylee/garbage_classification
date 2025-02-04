# garbage_classification/download_data.py

import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(dataset_name, download_path):
    """
    Downloads a dataset from Kaggle.

    Parameters:
        dataset_name (str): The Kaggle dataset identifier (e.g., 'username/dataset-name').
        download_path (str): The local directory to download the dataset to.
    """
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)

def main():
    print("Downloading dataset...")
    dataset = 'asdasdasasdas/garbage-classification'  # Replace with the actual dataset identifier
    download_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    os.makedirs(download_dir, exist_ok=True)
    download_dataset(dataset, download_dir)
    print("Download completed.")

if __name__ == "__main__":
    main()