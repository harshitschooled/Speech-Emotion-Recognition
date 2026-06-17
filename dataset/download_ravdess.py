import os
import urllib.request
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main():
    # Define URL and directories
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "raw")
    zip_path = os.path.join(base_dir, "ravdess_speech.zip")

    # Create target directories
    os.makedirs(raw_dir, exist_ok=True)

    print("RAVDESS Speech Dataset Downloader")
    print("==================================")
    print(f"Dataset URL: {url}")
    print(f"Target Directory: {raw_dir}")

    # Check if files already exist
    existing_wavs = []
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.endswith('.wav'):
                existing_wavs.append(file)
    
    if len(existing_wavs) >= 1440:
        print("RAVDESS dataset seems to be already downloaded and extracted.")
        print(f"Found {len(existing_wavs)} WAV files in {raw_dir}.")
        return

    # Download dataset
    if not os.path.exists(zip_path):
        print(f"Downloading RAVDESS dataset zip (approx 200MB)...")
        try:
            download_url(url, zip_path)
            print("\nDownload complete!")
        except Exception as e:
            print(f"\nError downloading file: {e}")
            print("Please download manually from: https://zenodo.org/record/1188976")
            print("Extract the zip contents directly into the 'dataset/raw/' folder.")
            return
    else:
        print(f"Zip file found at {zip_path}, skipping download.")

    # Extract dataset
    print(f"Extracting dataset to {raw_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files to show progress
            members = zip_ref.infolist()
            for member in tqdm(members, desc="Extracting"):
                zip_ref.extract(member, raw_dir)
        print("Extraction complete!")
        
        # Clean up zip file
        os.remove(zip_path)
        print("Removed temporary zip file.")
    except Exception as e:
        print(f"Error during extraction: {e}")

if __name__ == "__main__":
    main()
