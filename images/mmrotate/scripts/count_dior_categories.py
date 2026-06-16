#!/usr/bin/env python3
"""Count unique categories in DIOR-R labelTxt files."""
import glob
import os
from collections import Counter

LABEL_DIR = "/mnt/ht2_nas2/00-model/guantp/dino/mm_dino/data/DIOR-R/train/labelTxt"

def main():
    txt_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.txt")))
    print(f"Found {len(txt_files)} label files.\n")

    category_counter = Counter()

    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # Format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
                cls_name = parts[8]
                category_counter[cls_name] += 1

    print(f"{'Class Name':<30} {'Count':>8}")
    print("-" * 42)
    for cls_name, count in category_counter.most_common():
        print(f"{cls_name:<30} {count:>8}")

    print(f"\nTotal unique classes: {len(category_counter)}")
    print(f"Total instances: {sum(category_counter.values())}")

    # Print as a Python tuple for easy METAINFO comparison
    print("\n--- Unique classes (sorted alphabetically) ---")
    sorted_classes = sorted(category_counter.keys())
    for i, cls in enumerate(sorted_classes):
        print(f"  {i:2d}: '{cls}'")

    print(f"\n--- METAINFO 'classes' tuple (copy-paste ready) ---")
    print("(")
    for cls in sorted_classes:
        print(f"    '{cls}',")
    print(")")

    # Compare with current DIOR_DOTADataset METAINFO
    print("\n=== Comparison with current DIOR_DOTADataset METAINFO ===")
    current_metainfo = [
        'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
        'chimney', 'Expressway-Service-area', 'Expressway-toll-station',
        'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
        'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
        'windmill',
    ]
    current_set = set(current_metainfo)
    actual_set = set(category_counter.keys())

    missing_in_code = actual_set - current_set
    extra_in_code = current_set - actual_set

    if missing_in_code:
        print(f"\n  Classes in data but MISSING from METAINFO: {sorted(missing_in_code)}")
    else:
        print("\n  No classes missing from METAINFO.")

    if extra_in_code:
        print(f"  Classes in METAINFO but NOT in data: {sorted(extra_in_code)}")
    else:
        print("  No extra classes in METAINFO.")

    if not missing_in_code and not extra_in_code:
        print("  METAINFO matches data perfectly!")


if __name__ == "__main__":
    main()