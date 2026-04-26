import os
from pathlib import Path
import argparse

# Determine if we are running in Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Default base paths based on environment
if IN_COLAB:
    DEFAULT_BASE_DIR = Path('/content/data')
else:
    DEFAULT_BASE_DIR = Path(__file__).resolve().parent / 'data'

def get_base_dir(override_path=None):
    if override_path:
        return Path(override_path)
    return DEFAULT_BASE_DIR

# Using the default base dir to configure paths
BASE_DIR = get_base_dir()

COCO_VAL_IMAGES = BASE_DIR / 'val2017'
COCO_ANNOTATIONS = BASE_DIR / 'annotations'
MOCKUP_CSV_PATH = BASE_DIR / 'mockup_dataset.csv'
FULL_DATASET_CSV_PATH = BASE_DIR / 'hard_negative_dataset.csv'

# Ensure base data directory exists locally (if not in Colab)
if not IN_COLAB:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
