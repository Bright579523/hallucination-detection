import os
import sys

# Add parent directory to path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def download_coco():
    print(f"Downloading COCO val2017 to {config.BASE_DIR}...")
    # TODO: Implement download logic or instruct users to upload zip to drive

if __name__ == "__main__":
    download_coco()
