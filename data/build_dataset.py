import os
import sys
import argparse
import pandas as pd

# Add parent directory to path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def build_mockup_dataset(base_dir):
    print(f"Building mockup dataset using Base Directory: {base_dir}")
    
    # สร้างข้อมูลจำลอง (Mockup) 10 แถวเพื่อให้เห็นโครงสร้าง (Schema)
    mockup_data = [
        {"image_id": 1001, "image_path": "data/val2017/000000001001.jpg", "prompt": "a photo of dog", "true_label": 0, "supercategory": "animal", "negative_type": "positive"},
        {"image_id": 1002, "image_path": "data/val2017/000000001002.jpg", "prompt": "a photo of wolf", "true_label": 1, "supercategory": "animal", "negative_type": "hard"},
        {"image_id": 1003, "image_path": "data/val2017/000000001003.jpg", "prompt": "a photo of airplane", "true_label": 1, "supercategory": "animal", "negative_type": "easy"},
        {"image_id": 1004, "image_path": "data/val2017/000000001004.jpg", "prompt": "a photo of car", "true_label": 0, "supercategory": "vehicle", "negative_type": "positive"},
        {"image_id": 1005, "image_path": "data/val2017/000000001005.jpg", "prompt": "a photo of truck", "true_label": 1, "supercategory": "vehicle", "negative_type": "hard"},
        {"image_id": 1006, "image_path": "data/val2017/000000001006.jpg", "prompt": "a photo of cat", "true_label": 1, "supercategory": "vehicle", "negative_type": "easy"},
        {"image_id": 1007, "image_path": "data/val2017/000000001007.jpg", "prompt": "a photo of chair", "true_label": 0, "supercategory": "furniture", "negative_type": "positive"},
        {"image_id": 1008, "image_path": "data/val2017/000000001008.jpg", "prompt": "a photo of couch", "true_label": 1, "supercategory": "furniture", "negative_type": "hard"},
        {"image_id": 1009, "image_path": "data/val2017/000000001009.jpg", "prompt": "a photo of dog", "true_label": 1, "supercategory": "furniture", "negative_type": "easy"},
        {"image_id": 1010, "image_path": "data/val2017/000000001010.jpg", "prompt": "a photo of cat", "true_label": 0, "supercategory": "animal", "negative_type": "positive"},
    ]
    
    df = pd.DataFrame(mockup_data)
    mockup_path = base_dir / 'mockup_dataset.csv'
    
    # Save to CSV
    df.to_csv(mockup_path, index=False)
    print(f"Successfully saved mockup dataset ({len(df)} rows) to: {mockup_path}")
    print("\nPreview:")
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build mockup dataset.")
    parser.add_argument('--data_dir', type=str, default=None, help="Override default data directory path")
    args = parser.parse_args()
    
    base_dir = config.get_base_dir(args.data_dir)
    build_mockup_dataset(base_dir=base_dir)
