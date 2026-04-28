"""
download_coco.py — Person 1 (Data Engineer)

Downloads COCO val2017 images and annotations.
Designed to run on Google Colab (uses wget) or locally (uses urllib).

Usage:
  python data/download_coco.py                          # uses default paths
  python data/download_coco.py --data_dir /content/data  # override path
"""

import os
import sys
import argparse
import zipfile
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

COCO_URLS = {
    'val2017_images': 'http://images.cocodataset.org/zips/val2017.zip',
    'annotations':    'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
}


def download_and_extract(url, save_dir, filename):
    """Download a zip file and extract it to save_dir."""
    zip_path = save_dir / filename

    # Skip if already extracted
    extracted_name = filename.replace('.zip', '')
    if (save_dir / extracted_name).exists():
        print(f"  [SKIP] {extracted_name}/ already exists, skipping download.")
        return

    # Download
    print(f"  Downloading {filename} (~1GB for images, ~241MB for annotations)...")
    print(f"  URL: {url}")
    urllib.request.urlretrieve(url, str(zip_path))
    print(f"  Download complete: {zip_path}")

    # Extract
    print(f"  Extracting {filename}...")
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(save_dir))
    print(f"  Extraction complete.")

    # Cleanup zip
    os.remove(str(zip_path))
    print(f"  Cleaned up zip file.")


def download_coco(data_dir=None):
    """Download COCO val2017 images and annotations."""
    base_dir = config.get_base_dir(data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== COCO val2017 Downloader ===")
    print(f"Target directory: {base_dir}\n")

    # Download images
    print("[1/2] COCO val2017 images (5,000 images)")
    download_and_extract(
        COCO_URLS['val2017_images'], base_dir, 'val2017.zip'
    )

    # Download annotations
    print("\n[2/2] COCO val2017 annotations")
    download_and_extract(
        COCO_URLS['annotations'], base_dir, 'annotations_trainval2017.zip'
    )

    # Verify
    print("\n=== Verification ===")
    val_dir = base_dir / 'val2017'
    ann_dir = base_dir / 'annotations'

    if val_dir.exists():
        num_images = len([f for f in val_dir.iterdir() if f.suffix == '.jpg'])
        print(f"  Images:      {num_images} files in {val_dir}")
    else:
        print(f"  WARNING: {val_dir} not found!")

    if ann_dir.exists():
        ann_files = list(ann_dir.iterdir())
        print(f"  Annotations: {len(ann_files)} files in {ann_dir}")
    else:
        print(f"  WARNING: {ann_dir} not found!")

    print("\nDone! Ready to build dataset with: python data/build_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download COCO val2017 dataset.")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Override default data directory path")
    args = parser.parse_args()

    download_coco(data_dir=args.data_dir)
