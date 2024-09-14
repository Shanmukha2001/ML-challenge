import os
import pandas as pd
import requests
from tqdm import tqdm

def download_images(image_links, download_folder):
    """Download images from a list of image URLs and save them to a specified folder."""
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    for i, url in tqdm(enumerate(image_links), total=len(image_links), desc="Downloading images"):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check if the request was successful
            file_path = os.path.join(download_folder, f"image_{i + 1}.jpg")
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        except requests.RequestException as e:
            print(f"Failed to download {url}. Reason: {e}")

def main():
    DATASET_FOLDER = './'
    IMAGE_FOLDER = 'train_images'

    # Debugging: Print the current working directory
    print("Current Working Directory:", os.getcwd())

    # Check if the dataset folder exists
    if not os.path.exists(DATASET_FOLDER):
        print(f"Dataset folder does not exist: {DATASET_FOLDER}")
        return

    # Read dataset
    try:
        sample_test = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    except FileNotFoundError as e:
        print(e)
        print(f"Check if the file exists in the specified path: {os.path.join(DATASET_FOLDER, 'sample_test.csv')}")
        return

    # Download images
    download_images(sample_test['image_link'][40800:], IMAGE_FOLDER)

    # Verify download
    assert len(os.listdir(IMAGE_FOLDER)) > 0, "No images were downloaded"

if __name__ == "__main__":
    main()
