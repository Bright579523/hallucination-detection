"""
build_dataset.py — Person 1 (Data Engineer)

Reads COCO val2017 annotations and constructs a dataset CSV containing:
  - Positive samples:      image has object X + prompt "a photo of X"         (label=0)
  - Hard negative samples: image has object Y (same supercategory) + prompt X (label=1)
  - Easy negative samples: image has object Z (diff supercategory) + prompt X (label=1)

Usage:
  python data/build_dataset.py                          # uses default paths from config.py
  python data/build_dataset.py --data_dir /content/data  # override on Colab
"""

import os
import sys
import random
import argparse
import pandas as pd
from collections import defaultdict

# Add parent directory to path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Mockup dataset (for quick local testing without COCO files)
# ---------------------------------------------------------------------------
def build_mockup_dataset(base_dir):
    """Create a small 10-row mockup CSV so other modules can test their logic."""
    print(f"Building mockup dataset using Base Directory: {base_dir}")

    mockup_data = [
        {"image_id": 1001, "image_path": "val2017/000000001001.jpg", "prompt": "a photo of dog",      "true_label": 0, "supercategory": "animal",    "negative_type": "positive"},
        {"image_id": 1002, "image_path": "val2017/000000001002.jpg", "prompt": "a photo of wolf",     "true_label": 1, "supercategory": "animal",    "negative_type": "hard"},
        {"image_id": 1003, "image_path": "val2017/000000001003.jpg", "prompt": "a photo of airplane", "true_label": 1, "supercategory": "animal",    "negative_type": "easy"},
        {"image_id": 1004, "image_path": "val2017/000000001004.jpg", "prompt": "a photo of car",      "true_label": 0, "supercategory": "vehicle",   "negative_type": "positive"},
        {"image_id": 1005, "image_path": "val2017/000000001005.jpg", "prompt": "a photo of truck",    "true_label": 1, "supercategory": "vehicle",   "negative_type": "hard"},
        {"image_id": 1006, "image_path": "val2017/000000001006.jpg", "prompt": "a photo of cat",      "true_label": 1, "supercategory": "vehicle",   "negative_type": "easy"},
        {"image_id": 1007, "image_path": "val2017/000000001007.jpg", "prompt": "a photo of chair",    "true_label": 0, "supercategory": "furniture", "negative_type": "positive"},
        {"image_id": 1008, "image_path": "val2017/000000001008.jpg", "prompt": "a photo of couch",    "true_label": 1, "supercategory": "furniture", "negative_type": "hard"},
        {"image_id": 1009, "image_path": "val2017/000000001009.jpg", "prompt": "a photo of dog",      "true_label": 1, "supercategory": "furniture", "negative_type": "easy"},
        {"image_id": 1010, "image_path": "val2017/000000001010.jpg", "prompt": "a photo of cat",      "true_label": 0, "supercategory": "animal",    "negative_type": "positive"},
    ]

    df = pd.DataFrame(mockup_data)
    mockup_path = base_dir / 'mockup_dataset.csv'
    df.to_csv(mockup_path, index=False)
    print(f"Saved mockup dataset ({len(df)} rows) to: {mockup_path}")
    return df


# ---------------------------------------------------------------------------
# Full dataset builder (requires COCO annotation files)
# ---------------------------------------------------------------------------
def build_full_dataset(base_dir, max_positive=2000, max_hard=2000, max_easy=1000,
                       seed=42):
    """
    Build the full hard-negative dataset from COCO val2017 annotations.

    Parameters
    ----------
    base_dir : Path
        Root data directory containing val2017/ and annotations/ folders.
    max_positive : int
        Maximum number of positive samples to generate.
    max_hard : int
        Maximum number of hard negative samples to generate.
    max_easy : int
        Maximum number of easy negative samples to generate.
    seed : int
        Random seed for reproducibility.
    """
    random.seed(seed)

    # Import pycocotools (installed via requirements.txt)
    from pycocotools.coco import COCO

    ann_file = base_dir / 'annotations' / 'instances_val2017.json'
    if not ann_file.exists():
        print(f"ERROR: Annotation file not found at {ann_file}")
        print("Please download COCO val2017 annotations first.")
        return None

    print(f"Loading COCO annotations from: {ann_file}")
    coco = COCO(str(ann_file))

    # ------------------------------------------------------------------
    # Step 1: Build lookup tables
    # ------------------------------------------------------------------
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {c['id']: c['name'] for c in cats}
    cat_id_to_supercategory = {c['id']: c['supercategory'] for c in cats}

    # Group category IDs by supercategory
    # e.g. {'animal': [16, 17, 18, ...], 'vehicle': [1, 2, 3, ...]}
    supercategory_to_cat_ids = defaultdict(list)
    for c in cats:
        supercategory_to_cat_ids[c['supercategory']].append(c['id'])

    # For each image, find which category IDs are present
    img_ids = coco.getImgIds()
    img_to_cat_ids = {}
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_to_cat_ids[img_id] = set(a['category_id'] for a in anns)

    # Reverse mapping: category_id -> set of image_ids that contain it
    cat_id_to_img_ids = defaultdict(set)
    for img_id, cat_ids in img_to_cat_ids.items():
        for cat_id in cat_ids:
            cat_id_to_img_ids[cat_id].add(img_id)

    # ------------------------------------------------------------------
    # Step 2: Generate Positive Samples
    # ------------------------------------------------------------------
    print("Generating positive samples...")
    positive_rows = []
    all_cat_ids = list(cat_id_to_name.keys())
    random.shuffle(all_cat_ids)

    for cat_id in all_cat_ids:
        cat_name = cat_id_to_name[cat_id]
        supercat = cat_id_to_supercategory[cat_id]
        candidate_imgs = list(cat_id_to_img_ids[cat_id])
        random.shuffle(candidate_imgs)

        for img_id in candidate_imgs:
            if len(positive_rows) >= max_positive:
                break
            img_info = coco.loadImgs(img_id)[0]
            positive_rows.append({
                'image_id': img_id,
                'image_path': f"val2017/{img_info['file_name']}",
                'prompt': f"a photo of {cat_name}",
                'true_label': 0,
                'supercategory': supercat,
                'negative_type': 'positive',
            })
        if len(positive_rows) >= max_positive:
            break

    print(f"  -> {len(positive_rows)} positive samples")

    # ------------------------------------------------------------------
    # Step 3: Generate Hard Negative Samples
    #   Image has category Y, prompt asks for category X,
    #   where X and Y share the same supercategory but X != Y
    # ------------------------------------------------------------------
    print("Generating hard negative samples...")
    hard_rows = []

    for cat_id in all_cat_ids:
        cat_name = cat_id_to_name[cat_id]
        supercat = cat_id_to_supercategory[cat_id]

        # Find sibling categories in the same supercategory
        sibling_cat_ids = [
            c for c in supercategory_to_cat_ids[supercat] if c != cat_id
        ]
        if not sibling_cat_ids:
            continue  # No siblings available for hard negatives

        # Find images that contain a sibling but NOT this category
        for sibling_id in sibling_cat_ids:
            imgs_with_sibling = cat_id_to_img_ids[sibling_id]
            imgs_without_target = imgs_with_sibling - cat_id_to_img_ids[cat_id]
            candidate_imgs = list(imgs_without_target)
            random.shuffle(candidate_imgs)

            for img_id in candidate_imgs[:5]:  # max 5 per sibling pair
                if len(hard_rows) >= max_hard:
                    break
                img_info = coco.loadImgs(img_id)[0]
                hard_rows.append({
                    'image_id': img_id,
                    'image_path': f"val2017/{img_info['file_name']}",
                    'prompt': f"a photo of {cat_name}",
                    'true_label': 1,
                    'supercategory': supercat,
                    'negative_type': 'hard',
                })
            if len(hard_rows) >= max_hard:
                break
        if len(hard_rows) >= max_hard:
            break

    print(f"  -> {len(hard_rows)} hard negative samples")

    # ------------------------------------------------------------------
    # Step 4: Generate Easy Negative Samples
    #   Image has category Z from a DIFFERENT supercategory,
    #   prompt asks for category X
    # ------------------------------------------------------------------
    print("Generating easy negative samples...")
    easy_rows = []
    all_supercategories = list(supercategory_to_cat_ids.keys())

    for cat_id in all_cat_ids:
        cat_name = cat_id_to_name[cat_id]
        supercat = cat_id_to_supercategory[cat_id]

        # Pick a random different supercategory
        other_supercats = [s for s in all_supercategories if s != supercat]
        if not other_supercats:
            continue

        other_supercat = random.choice(other_supercats)
        other_cat_ids = supercategory_to_cat_ids[other_supercat]

        # Find images from the other supercategory that do NOT contain target
        for other_cat_id in other_cat_ids:
            imgs_with_other = cat_id_to_img_ids[other_cat_id]
            imgs_without_target = imgs_with_other - cat_id_to_img_ids[cat_id]
            candidate_imgs = list(imgs_without_target)
            random.shuffle(candidate_imgs)

            for img_id in candidate_imgs[:3]:  # max 3 per pair
                if len(easy_rows) >= max_easy:
                    break
                img_info = coco.loadImgs(img_id)[0]
                easy_rows.append({
                    'image_id': img_id,
                    'image_path': f"val2017/{img_info['file_name']}",
                    'prompt': f"a photo of {cat_name}",
                    'true_label': 1,
                    'supercategory': supercat,
                    'negative_type': 'easy',
                })
            if len(easy_rows) >= max_easy:
                break
        if len(easy_rows) >= max_easy:
            break

    print(f"  -> {len(easy_rows)} easy negative samples")

    # ------------------------------------------------------------------
    # Step 5: Combine, shuffle, and save
    # ------------------------------------------------------------------
    all_rows = positive_rows + hard_rows + easy_rows
    random.shuffle(all_rows)
    df = pd.DataFrame(all_rows)

    output_path = base_dir / 'hard_negative_dataset.csv'
    df.to_csv(output_path, index=False)

    print(f"\n=== Dataset Summary ===")
    print(f"Total samples  : {len(df)}")
    print(f"  Positive     : {len(positive_rows)}")
    print(f"  Hard negative: {len(hard_rows)}")
    print(f"  Easy negative: {len(easy_rows)}")
    print(f"Saved to       : {output_path}")
    print(f"\nLabel distribution:")
    print(df['negative_type'].value_counts())
    print(f"\nSupercategory distribution:")
    print(df['supercategory'].value_counts())
    print(f"\nPreview (first 10 rows):")
    print(df.head(10).to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build hallucination detection dataset from COCO val2017."
    )
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Override default data directory path")
    parser.add_argument('--mode', type=str, default='full',
                        choices=['mockup', 'full'],
                        help="'mockup' for 10-row test CSV, 'full' for real COCO dataset")
    parser.add_argument('--max_positive', type=int, default=2000,
                        help="Max positive samples (default: 2000)")
    parser.add_argument('--max_hard', type=int, default=2000,
                        help="Max hard negative samples (default: 2000)")
    parser.add_argument('--max_easy', type=int, default=1000,
                        help="Max easy negative samples (default: 1000)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    base_dir = config.get_base_dir(args.data_dir)

    if args.mode == 'mockup':
        build_mockup_dataset(base_dir)
    else:
        build_full_dataset(
            base_dir,
            max_positive=args.max_positive,
            max_hard=args.max_hard,
            max_easy=args.max_easy,
            seed=args.seed,
        )
