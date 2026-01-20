#!/usr/bin/env python3
"""
Dataset Comparison Script - Cotton Weed Detection

Compares the competition dataset with the original dataset to detect amendments.
Checks for:
- Missing or added images
- Image content changes (via hash comparison)
- Label modifications (if convertible to YOLO format)
- Metadata differences

Usage:
    python compare_datasets.py
"""

from pathlib import Path
import hashlib
import json
from collections import defaultdict
import cv2
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get the script's directory to make paths relative to it
SCRIPT_DIR = Path(__file__).resolve().parent

# Original dataset paths
ORIGINAL_BASE = SCRIPT_DIR / "cottonweeddet3-DatasetNinja" / "ds"
ORIGINAL_IMAGES = ORIGINAL_BASE / "img"
ORIGINAL_ANNOTATIONS = ORIGINAL_BASE / "ann"

# Competition dataset paths
COMPETITION_BASE = SCRIPT_DIR / "cotton_weed_competition_dataset"
COMPETITION_TRAIN = COMPETITION_BASE / "train"
COMPETITION_VAL = COMPETITION_BASE / "val"
COMPETITION_TEST = COMPETITION_BASE / "test"

# Output
OUTPUT_REPORT = SCRIPT_DIR / "dataset_comparison_report.json"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Class mapping from classTitle to class_id
CLASS_MAPPING = {
    "carpetweed": 0,
    "morningglory": 1,
    "palmer_amaranth": 2,
    "palmer amaranth": 2,  # Handle space in name
    "palmeramaranth": 2  # Handle possible variations
}

def get_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"Error hashing {filepath}: {e}")
        return None


def get_image_files(directory):
    """Get all image files from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    if not directory.exists():
        return {}
    
    files = {}
    for ext in image_extensions:
        for img_path in directory.glob(f"*{ext}"):
            files[img_path.stem] = img_path
        for img_path in directory.glob(f"*{ext.upper()}"):
            files[img_path.stem] = img_path
    
    return files


def get_label_files(directory):
    """Get all label files from a directory."""
    if not directory.exists():
        return {}
    
    files = {}
    for label_path in directory.glob("*.txt"):
        files[label_path.stem] = label_path
    
    return files


def compare_images(img1_path, img2_path):
    """Compare two images for similarity."""
    try:
        # First check file hash
        hash1 = get_file_hash(img1_path)
        hash2 = get_file_hash(img2_path)
        
        if hash1 == hash2:
            return {"identical": True, "hash_match": True}
        
        # If hashes differ, check visual similarity
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        if img1 is None or img2 is None:
            return {"identical": False, "error": "Could not load images"}
        
        # Check dimensions
        same_size = img1.shape == img2.shape
        
        # Calculate structural similarity if same size
        if same_size:
            diff = cv2.absdiff(img1, img2)
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            
            return {
                "identical": False,
                "hash_match": False,
                "same_size": True,
                "mean_pixel_diff": float(mean_diff),
                "max_pixel_diff": float(max_diff),
                "visually_identical": mean_diff < 1.0
            }
        else:
            return {
                "identical": False,
                "hash_match": False,
                "same_size": False,
                "original_size": img1.shape,
                "competition_size": img2.shape
            }
            
    except Exception as e:
        return {"identical": False, "error": str(e)}


def read_yolo_label(label_path):
    """Read YOLO format label file."""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append({
                    'class': int(parts[0]),
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'w': float(parts[3]),
                    'h': float(parts[4])
                })
        return boxes
    except Exception as e:
        print(f"Error reading label {label_path}: {e}")
        return None


def read_json_annotation(json_path):
    """Read original JSON annotation file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading annotation {json_path}: {e}")
        return None


def json_to_yolo_boxes(json_data):
    """Convert JSON annotation to YOLO format boxes.
    
    Args:
        json_data: Parsed JSON annotation
        
    Returns:
        List of boxes in YOLO format (normalized coordinates)
    """
    if not json_data or 'objects' not in json_data or 'size' not in json_data:
        return None
    
    img_height = json_data['size']['height']
    img_width = json_data['size']['width']
    
    boxes = []
    for obj in json_data['objects']:
        # Get class
        class_title = obj.get('classTitle', '').lower()
        if class_title not in CLASS_MAPPING:
            print(f"Warning: Unknown class '{class_title}', skipping")
            continue
        
        class_id = CLASS_MAPPING[class_title]
        
        # Get bounding box coordinates
        if 'points' not in obj or 'exterior' not in obj['points']:
            continue
        
        exterior = obj['points']['exterior']
        if len(exterior) < 2:
            continue
        
        # Extract corner points [top_left, bottom_right]
        x1, y1 = exterior[0]
        x2, y2 = exterior[1]
        
        # Calculate YOLO format (center_x, center_y, width, height) normalized
        bbox_width = abs(x2 - x1)
        bbox_height = abs(y2 - y1)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # Normalize by image dimensions
        x_norm = center_x / img_width
        y_norm = center_y / img_height
        w_norm = bbox_width / img_width
        h_norm = bbox_height / img_height
        
        boxes.append({
            'class': class_id,
            'x': x_norm,
            'y': y_norm,
            'w': w_norm,
            'h': h_norm
        })
    
    return boxes


def compare_labels(label1_path, label2_path, tolerance=0.01):
    """Compare two YOLO label files.
    
    Args:
        label1_path: Path to first label (can be JSON or YOLO)
        label2_path: Path to second label (YOLO format)
        tolerance: Tolerance for floating point comparison (default 0.01 = 1% of image size)
    """
    # Read labels based on file extension
    if str(label1_path).endswith('.json'):
        json_data = read_json_annotation(label1_path)
        boxes1 = json_to_yolo_boxes(json_data)
    else:
        boxes1 = read_yolo_label(label1_path)
    
    boxes2 = read_yolo_label(label2_path)
    
    if boxes1 is None or boxes2 is None:
        return {"error": "Could not read labels"}
    
    if len(boxes1) != len(boxes2):
        return {
            "identical": False,
            "num_boxes_original": len(boxes1),
            "num_boxes_competition": len(boxes2),
            "box_count_changed": True
        }
    
    # Compare boxes (order-independent)
    boxes1_sorted = sorted(boxes1, key=lambda x: (x['class'], x['x'], x['y']))
    boxes2_sorted = sorted(boxes2, key=lambda x: (x['class'], x['x'], x['y']))
    
    # Check if boxes are identical within tolerance
    identical = True
    differences = []
    
    for b1, b2 in zip(boxes1_sorted, boxes2_sorted):
        if b1['class'] != b2['class']:
            identical = False
            differences.append({
                'type': 'class_mismatch',
                'original': b1,
                'competition': b2
            })
        elif (abs(b1['x'] - b2['x']) > tolerance or
              abs(b1['y'] - b2['y']) > tolerance or
              abs(b1['w'] - b2['w']) > tolerance or
              abs(b1['h'] - b2['h']) > tolerance):
            identical = False
            differences.append({
                'type': 'coordinate_mismatch',
                'original': b1,
                'competition': b2,
                'diff_x': abs(b1['x'] - b2['x']),
                'diff_y': abs(b1['y'] - b2['y']),
                'diff_w': abs(b1['w'] - b2['w']),
                'diff_h': abs(b1['h'] - b2['h'])
            })
    
    result = {
        "identical": identical,
        "num_boxes": len(boxes1),
        "box_count_changed": False
    }
    
    if not identical:
        result["differences"] = differences
        result["num_differences"] = len(differences)
    
    return result


# ============================================================================
# MAIN COMPARISON LOGIC
# ============================================================================

def main():
    """Compare original and competition datasets."""
    
    print("=" * 70)
    print("Dataset Comparison Tool")
    print("=" * 70)
    
    # Check paths exist
    if not ORIGINAL_IMAGES.exists():
        print(f"\n❌ ERROR: Original images not found at {ORIGINAL_IMAGES}")
        return
    
    print(f"\n✅ Original dataset: {ORIGINAL_BASE}")
    print(f"✅ Competition dataset: {COMPETITION_BASE}")
    
    # Collect all image files
    print("\n" + "=" * 70)
    print("Step 1: Collecting Files")
    print("=" * 70)
    
    original_images = get_image_files(ORIGINAL_IMAGES)
    print(f"  Original images: {len(original_images)}")
    
    competition_train_images = get_image_files(COMPETITION_TRAIN / "images")
    competition_val_images = get_image_files(COMPETITION_VAL / "images")
    competition_test_images = get_image_files(COMPETITION_TEST / "images")
    
    competition_all_images = {
        **competition_train_images,
        **competition_val_images,
        **competition_test_images
    }
    
    print(f"  Competition train: {len(competition_train_images)}")
    print(f"  Competition val: {len(competition_val_images)}")
    print(f"  Competition test: {len(competition_test_images)}")
    print(f"  Competition total: {len(competition_all_images)}")
    
    # Find differences in image sets
    print("\n" + "=" * 70)
    print("Step 2: Comparing Image Sets")
    print("=" * 70)
    
    original_names = set(original_images.keys())
    competition_names = set(competition_all_images.keys())
    
    missing_in_competition = original_names - competition_names
    added_in_competition = competition_names - original_names
    common_images = original_names & competition_names
    
    print(f"\n  Common images: {len(common_images)}")
    print(f"  Missing in competition: {len(missing_in_competition)}")
    print(f"  Added in competition: {len(added_in_competition)}")
    
    if missing_in_competition:
        print(f"\n  Missing images (first 10): {list(missing_in_competition)[:10]}")
    if added_in_competition:
        print(f"  Added images (first 10): {list(added_in_competition)[:10]}")
    
    # Compare common images
    print("\n" + "=" * 70)
    print("Step 3: Comparing Image Content")
    print("=" * 70)
    
    image_comparisons = {}
    modified_images = []
    
    print(f"\n  Comparing {len(common_images)} common images...")
    
    for i, img_name in enumerate(sorted(common_images), 1):
        if i % 100 == 0:
            print(f"    Progress: {i}/{len(common_images)}")
        
        original_path = original_images[img_name]
        competition_path = competition_all_images[img_name]
        
        comparison = compare_images(original_path, competition_path)
        
        if not comparison.get("identical", False):
            modified_images.append(img_name)
            image_comparisons[img_name] = comparison
    
    print(f"\n  ✅ Identical images: {len(common_images) - len(modified_images)}")
    print(f"  ⚠️  Modified images: {len(modified_images)}")
    
    # Compare labels (if available in YOLO format)
    print("\n" + "=" * 70)
    print("Step 4: Comparing Labels")
    print("=" * 70)
    
    # Get original annotations (JSON format)
    original_annotations = {}
    for json_path in ORIGINAL_ANNOTATIONS.glob("*.json"):
        # Extract image name without extension
        img_name = json_path.stem.replace('.jpg', '').replace('.png', '')
        original_annotations[img_name] = json_path
    
    print(f"  Original annotations (JSON): {len(original_annotations)}")
    
    competition_train_labels = get_label_files(COMPETITION_TRAIN / "labels")
    competition_val_labels = get_label_files(COMPETITION_VAL / "labels")
    
    all_competition_labels = {**competition_train_labels, **competition_val_labels}
    
    print(f"  Competition labels (YOLO): {len(all_competition_labels)}")
    
    # Find common images that have labels in both datasets
    original_labeled = set(original_annotations.keys())
    competition_labeled = set(all_competition_labels.keys())
    common_labeled = original_labeled & competition_labeled
    
    print(f"  Common labeled images: {len(common_labeled)}")
    print(f"  Only in original: {len(original_labeled - competition_labeled)}")
    print(f"  Only in competition: {len(competition_labeled - original_labeled)}")
    
    # Compare labels for common images
    label_comparisons = {}
    identical_labels = 0
    modified_labels = []
    conversion_errors = []
    
    print(f"\n  Comparing {len(common_labeled)} labeled images...")
    
    for i, img_name in enumerate(sorted(common_labeled), 1):
        if i % 50 == 0:
            print(f"    Progress: {i}/{len(common_labeled)}")
        
        original_json = original_annotations[img_name]
        competition_label = all_competition_labels[img_name]
        
        comparison = compare_labels(original_json, competition_label, tolerance=0.1)
        
        if "error" in comparison:
            conversion_errors.append(img_name)
            label_comparisons[img_name] = comparison
        elif comparison.get("identical", False):
            identical_labels += 1
        else:
            modified_labels.append(img_name)
            label_comparisons[img_name] = comparison
    
    print(f"\n  ✅ Identical labels: {identical_labels}")
    print(f"  ⚠️  Modified labels: {len(modified_labels)}")
    print(f"  ❌ Conversion errors: {len(conversion_errors)}")
    
    # Analyze types of modifications
    box_count_changes = []
    coordinate_changes = []
    class_changes = []
    max_coord_diffs = []
    
    for img_name in modified_labels:
        comp = label_comparisons[img_name]
        if comp.get("box_count_changed"):
            box_count_changes.append(img_name)
        elif "differences" in comp:
            has_coord_change = False
            has_class_change = False
            max_diff = 0.0
            for diff in comp["differences"]:
                if diff["type"] == "class_mismatch":
                    has_class_change = True
                elif diff["type"] == "coordinate_mismatch":
                    has_coord_change = True
                    # Track maximum coordinate difference for this image
                    curr_max = max(diff['diff_x'], diff['diff_y'], diff['diff_w'], diff['diff_h'])
                    max_diff = max(max_diff, curr_max)
            if has_class_change:
                class_changes.append(img_name)
            if has_coord_change:
                coordinate_changes.append(img_name)
                max_coord_diffs.append(max_diff)
    
    print(f"\n  Modification types:")
    print(f"    - Box count changed: {len(box_count_changes)}")
    print(f"    - Class changes: {len(class_changes)}")
    print(f"    - Coordinate changes: {len(coordinate_changes)}")
    
    if max_coord_diffs:
        print(f"\n  Coordinate difference statistics:")
        print(f"    - Min: {min(max_coord_diffs):.6f}")
        print(f"    - Max: {max(max_coord_diffs):.6f}")
        print(f"    - Mean: {sum(max_coord_diffs)/len(max_coord_diffs):.6f}")
        print(f"    - Median: {sorted(max_coord_diffs)[len(max_coord_diffs)//2]:.6f}")
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("Summary Report")
    print("=" * 70)
    
    report = {
        "dataset_statistics": {
            "original_images": len(original_images),
            "competition_train": len(competition_train_images),
            "competition_val": len(competition_val_images),
            "competition_test": len(competition_test_images),
            "competition_total": len(competition_all_images)
        },
        "image_set_differences": {
            "common_images": len(common_images),
            "missing_in_competition": len(missing_in_competition),
            "added_in_competition": len(added_in_competition),
            "missing_list": sorted(list(missing_in_competition)),
            "added_list": sorted(list(added_in_competition))
        },
        "image_content_changes": {
            "identical_images": len(common_images) - len(modified_images),
            "modified_images": len(modified_images),
            "modified_list": sorted(modified_images),
            "modifications": image_comparisons
        },
        "label_statistics": {
            "original_annotations": len(original_annotations),
            "competition_labels": len(all_competition_labels),
            "common_labeled_images": len(common_labeled),
            "only_in_original": len(original_labeled - competition_labeled),
            "only_in_competition": len(competition_labeled - original_labeled)
        },
        "label_comparisons": {
            "identical_labels": identical_labels,
            "modified_labels": len(modified_labels),
            "conversion_errors": len(conversion_errors),
            "modification_types": {
                "box_count_changed": len(box_count_changes),
                "class_changes": len(class_changes),
                "coordinate_changes": len(coordinate_changes)
            },
            "box_count_changes_list": sorted(box_count_changes),
            "class_changes_list": sorted(class_changes),
            "coordinate_changes_list": sorted(coordinate_changes),
            "conversion_errors_list": sorted(conversion_errors),
            "detailed_comparisons": label_comparisons
        }
    }
    
    # Save report
    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Detailed report saved to: {OUTPUT_REPORT}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Key Findings")
    print("=" * 70)
    
    no_image_changes = (len(missing_in_competition) == 0 and 
                        len(added_in_competition) == 0 and 
                        len(modified_images) == 0)
    no_label_changes = (len(modified_labels) == 0 and len(conversion_errors) == 0)
    
    if no_image_changes and no_label_changes:
        print("\n✅ NO CHANGES DETECTED")
        print("   The competition dataset appears identical to the original")
    else:
        print(f"\n⚠️  CHANGES DETECTED:")
        
        # Image changes
        if not no_image_changes:
            print(f"\n   IMAGE CHANGES:")
            if missing_in_competition:
                print(f"   - {len(missing_in_competition)} images removed from original")
            if added_in_competition:
                print(f"   - {len(added_in_competition)} new images added")
            if modified_images:
                print(f"   - {len(modified_images)} images modified")
                
                # Show examples of modifications
                print(f"\n   Modified images (first 5):")
                for img_name in sorted(modified_images)[:5]:
                    comparison = image_comparisons[img_name]
                    print(f"     • {img_name}:")
                    if not comparison.get("same_size", True):
                        print(f"       - Size changed: {comparison.get('original_size')} → {comparison.get('competition_size')}")
                    elif comparison.get("mean_pixel_diff"):
                        print(f"       - Pixel difference: mean={comparison['mean_pixel_diff']:.2f}, max={comparison['max_pixel_diff']:.2f}")
        
        # Label changes
        if not no_label_changes:
            print(f"\n   LABEL CHANGES:")
            print(f"   - Identical labels: {identical_labels}/{len(common_labeled)}")
            print(f"   - Modified labels: {len(modified_labels)}")
            
            if box_count_changes:
                print(f"     • Box count changed: {len(box_count_changes)}")
                print(f"       Examples: {', '.join(sorted(box_count_changes)[:3])}")
            
            if class_changes:
                print(f"     • Class changes: {len(class_changes)}")
                print(f"       Examples: {', '.join(sorted(class_changes)[:3])}")
            
            if coordinate_changes:
                print(f"     • Coordinate changes: {len(coordinate_changes)}")
                print(f"       Examples: {', '.join(sorted(coordinate_changes)[:3])}")
            
            if conversion_errors:
                print(f"     • Conversion errors: {len(conversion_errors)}")
                print(f"       Examples: {', '.join(sorted(conversion_errors)[:3])}")
            
            # Show detailed example of first label change
            if modified_labels and modified_labels[0] in label_comparisons:
                print(f"\n   Example detailed comparison ({modified_labels[0]}):")
                comp = label_comparisons[modified_labels[0]]
                if comp.get("box_count_changed"):
                    print(f"     Original boxes: {comp['num_boxes_original']}")
                    print(f"     Competition boxes: {comp['num_boxes_competition']}")
                elif "differences" in comp and len(comp["differences"]) > 0:
                    diff = comp["differences"][0]
                    print(f"     Type: {diff['type']}")
                    print(f"     Original: class={diff['original']['class']}, x={diff['original']['x']:.4f}, y={diff['original']['y']:.4f}")
                    print(f"     Competition: class={diff['competition']['class']}, x={diff['competition']['x']:.4f}, y={diff['competition']['y']:.4f}")
                    if 'diff_x' in diff:
                        print(f"     Max difference: {max(diff['diff_x'], diff['diff_y'], diff['diff_w'], diff['diff_h']):.6f}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
