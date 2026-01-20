#!/usr/bin/env python3
"""
Generate Summary CSV from Dataset Comparison

Creates a comprehensive CSV file showing all images with:
- Train/Val/Test split information
- Label status and changes
- Image availability in both datasets
"""

import json
import pandas as pd
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_FILE = SCRIPT_DIR / "dataset_comparison_report.json"
OUTPUT_CSV = SCRIPT_DIR / "dataset_comparison_summary.csv"

# Competition dataset paths
COMPETITION_BASE = SCRIPT_DIR / "cotton_weed_competition_dataset"
COMPETITION_TRAIN = COMPETITION_BASE / "train"
COMPETITION_VAL = COMPETITION_BASE / "val"
COMPETITION_TEST = COMPETITION_BASE / "test"


def get_image_split(img_name, train_images, val_images, test_images):
    """Determine which split an image belongs to."""
    if img_name in train_images:
        return 'train'
    elif img_name in val_images:
        return 'val'
    elif img_name in test_images:
        return 'test'
    else:
        return 'not_in_competition'


def get_image_files(directory):
    """Get all image files from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    if not directory.exists():
        return set()
    
    files = set()
    for ext in image_extensions:
        for img_path in directory.glob(f"*{ext}"):
            files.add(img_path.stem)
        for img_path in directory.glob(f"*{ext.upper()}"):
            files.add(img_path.stem)
    
    return files


def main():
    """Generate summary CSV from comparison report."""
    
    print("=" * 70)
    print("Dataset Comparison Summary CSV Generator")
    print("=" * 70)
    
    # Load the comparison report
    print(f"\nLoading report: {REPORT_FILE}")
    with open(REPORT_FILE, 'r') as f:
        report = json.load(f)
    
    # Get split information
    print("Determining train/val/test splits...")
    train_images = get_image_files(COMPETITION_TRAIN / "images")
    val_images = get_image_files(COMPETITION_VAL / "images")
    test_images = get_image_files(COMPETITION_TEST / "images")
    
    print(f"  Train: {len(train_images)}")
    print(f"  Val: {len(val_images)}")
    print(f"  Test: {len(test_images)}")
    
    # Get all unique image names from original and competition
    all_images = set()
    
    # From report
    all_images.update(report['image_set_differences']['missing_list'])
    all_images.update(report['image_set_differences']['added_list'])
    
    # From splits
    all_images.update(train_images)
    all_images.update(val_images)
    all_images.update(test_images)
    
    print(f"\nTotal unique images: {len(all_images)}")
    
    # Create data rows
    data = []
    
    for img_name in sorted(all_images):
        # Determine split
        split = get_image_split(img_name, train_images, val_images, test_images)
        
        # Check if in original dataset
        in_original = img_name not in report['image_set_differences']['added_list']
        
        # Check if in competition dataset
        in_competition = img_name not in report['image_set_differences']['missing_list']
        
        # Check if image content was modified
        image_modified = img_name in report['image_content_changes']['modified_list']
        
        # Check label information
        has_competition_label = False
        has_original_label = False
        label_status = 'no_label'
        label_change_type = None
        num_boxes_original = None
        num_boxes_competition = None
        
        if split in ['train', 'val']:
            has_competition_label = True
            
            # Check if it has original label
            if in_original:
                # Check if it was in the common labeled set
                label_comparisons = report['label_comparisons']['detailed_comparisons']
                
                if img_name in label_comparisons:
                    has_original_label = True
                    comp = label_comparisons[img_name]
                    
                    if 'error' in comp:
                        label_status = 'conversion_error'
                    elif comp.get('identical', False):
                        label_status = 'identical'
                    elif comp.get('box_count_changed', False):
                        label_status = 'modified'
                        label_change_type = 'box_count_changed'
                        num_boxes_original = comp.get('num_boxes_original')
                        num_boxes_competition = comp.get('num_boxes_competition')
                    elif 'differences' in comp:
                        label_status = 'modified'
                        # Determine change type
                        for diff in comp.get('differences', []):
                            if diff['type'] == 'class_mismatch':
                                label_change_type = 'class_changed'
                                break
                            elif diff['type'] == 'coordinate_mismatch':
                                label_change_type = 'coordinates_changed'
                        num_boxes_original = comp.get('num_boxes')
                        num_boxes_competition = comp.get('num_boxes')
                else:
                    # Has competition label but not in comparison (no original label)
                    label_status = 'competition_only'
                    has_original_label = False
        elif split == 'test':
            # Test images don't have labels in competition
            has_competition_label = False
            label_status = 'test_no_label'
        
        # Create row
        row = {
            'image_name': img_name,
            'split': split,
            'in_original': in_original,
            'in_competition': in_competition,
            'image_modified': image_modified,
            'has_original_label': has_original_label,
            'has_competition_label': has_competition_label,
            'label_status': label_status,
            'label_change_type': label_change_type,
            'num_boxes_original': num_boxes_original,
            'num_boxes_competition': num_boxes_competition
        }
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    print(f"\nSaving summary to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    
    print(f"\n{'Dataset Split':<25} {'Count':>10}")
    print("-" * 37)
    for split in ['train', 'val', 'test', 'not_in_competition']:
        count = len(df[df['split'] == split])
        print(f"{split:<25} {count:>10}")
    
    print(f"\n{'Label Status':<25} {'Count':>10}")
    print("-" * 37)
    for status in df['label_status'].unique():
        count = len(df[df['label_status'] == status])
        print(f"{status:<25} {count:>10}")
    
    print(f"\n{'Label Change Type':<25} {'Count':>10}")
    print("-" * 37)
    for change_type in df['label_change_type'].dropna().unique():
        count = len(df[df['label_change_type'] == change_type])
        print(f"{change_type:<25} {count:>10}")
    
    print(f"\n{'Image Status':<25} {'Count':>10}")
    print("-" * 37)
    print(f"{'In both datasets':<25} {len(df[df['in_original'] & df['in_competition']]):>10}")
    print(f"{'Only in original':<25} {len(df[df['in_original'] & ~df['in_competition']]):>10}")
    print(f"{'Only in competition':<25} {len(df[~df['in_original'] & df['in_competition']]):>10}")
    
    print("\n" + "=" * 70)
    print("CSV Generation Complete")
    print("=" * 70)
    print(f"\nTotal rows: {len(df)}")
    print(f"Output file: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
