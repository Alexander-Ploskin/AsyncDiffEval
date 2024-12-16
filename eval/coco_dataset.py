import os
import requests
import zipfile


os.makedirs('coco/images', exist_ok=True)

urls = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/zips/test2017.zip",
    "http://images.cocodataset.org/zips/unlabeled2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
]

def download_file(url):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

for url in urls:
    print(f"Downloading {url}...")
    zip_file = download_file(url)
    
    print(f"Unzipping {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('coco/images')
    
    os.remove(zip_file)
    print(f"Removed {zip_file}")

print("All files downloaded and extracted successfully.")
