#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create input CSV from PlantVillage segmented dataset.

Scans the dataset/plantvillage/data/PlantVillage-Dataset/raw/segmented directory
and creates a CSV with image paths and labels.

Column names match the original format:
- nom_plante, nom_maladie, Est_Saine, Image_Path
- width_img, height_img, is_black, md5
"""

from pathlib import Path
import pandas as pd
from typing import List, Dict
import argparse
import hashlib
from PIL import Image, ImageStat
from tqdm import tqdm


def compute_image_metadata(img_path: Path) -> Dict:
    """
    Compute metadata for an image.
    
    Args:
        img_path: Path to the image file
    
    Returns:
        Dictionary with width_img, height_img, is_black, md5
    """
    try:
        # Open image to get dimensions
        with Image.open(img_path) as img:
            width, height = img.size
            
            # Check if image is black
            stat = ImageStat.Stat(img.convert('L'))
            is_black = stat.mean[0] < 10
        
        # Compute MD5 hash
        with open(img_path, 'rb') as f:
            md5_hash = hashlib.md5(f.read()).hexdigest()
        
        return {
            'width_img': width,
            'height_img': height,
            'is_black': is_black,
            'md5': md5_hash
        }
    except Exception as e:
        # Return default values on error
        return {
            'width_img': 0,
            'height_img': 0,
            'is_black': False,
            'md5': ''
        }


def scan_segmented_dataset(base_path: Path) -> List[Dict[str, str]]:
    """
    Scan the segmented dataset directory and collect image information.
    
    Args:
        base_path: Path to the segmented directory
    
    Returns:
        List of dictionaries with image information
    """
    records = []
    
    # Count total images first for progress bar
    class_dirs = [d for d in sorted(base_path.iterdir()) if d.is_dir()]
    total_images = sum(len(list(d.glob("*.jpg")) + list(d.glob("*.JPG"))) for d in class_dirs)
    
    # Iterate through class directories with progress bar
    with tqdm(total=total_images, desc="Scanning images", unit="img") as pbar:
        for class_dir in class_dirs:
            class_name = class_dir.name
            
            # Parse class name to extract plant and disease
            # Format: "Plant___Disease" or "Plant___healthy"
            parts = class_name.split("___")
            if len(parts) == 2:
                plant = parts[0].strip()
                disease = parts[1].strip()
                is_healthy = disease.lower() == "healthy"
            else:
                plant = class_name
                disease = "unknown"
                is_healthy = False
            
            # Iterate through images in this class
            for img_path in sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.JPG")):
                # Compute image metadata
                metadata = compute_image_metadata(img_path)
                
                records.append({
                    "nom_plante": plant,
                    "nom_maladie": disease,
                    "Est_Saine": is_healthy,
                    "Image_Path": str(img_path),
                    "width_img": metadata['width_img'],
                    "height_img": metadata['height_img'],
                    "is_black": metadata['is_black'],
                    "md5": metadata['md5'],
                })
                pbar.update(1)
    
    return records


def main():
    parser = argparse.ArgumentParser(description="Create input CSV from PlantVillage segmented dataset")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/plantvillage/data/PlantVillage-Dataset/raw/segmented",
        help="Path to segmented dataset directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/plantvillage/csv/clean_data_plantvillage_segmented_all.csv",
        help="Output CSV file path"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    project_root = Path(__file__).parent.parent.parent
    dataset_path = project_root / args.dataset_path
    output_path = project_root / args.output
    
    print(f"Scanning dataset: {dataset_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Scan dataset
    records = scan_segmented_dataset(dataset_path)
    
    # Count unique classes (plant + disease combinations)
    unique_classes = len(set((r['nom_plante'], r['nom_maladie']) for r in records))
    print(f"\nFound {len(records)} images across {unique_classes} classes")
    
    # Create DataFrame with specific column order matching old CSV
    df = pd.DataFrame(records)
    
    # Ensure column order matches old CSV
    column_order = [
        'nom_plante', 'nom_maladie', 'Est_Saine', 'Image_Path',
        'width_img', 'height_img', 'is_black', 'md5'
    ]
    df = df[column_order]
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    print(f"Writing to: {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\nDataset summary:")
    print(f"  Total images: {len(df)}")
    print(f"  Unique plant-disease combinations: {unique_classes}")
    print(f"  Healthy images: {df['Est_Saine'].sum()}")
    print(f"  Diseased images: {(~df['Est_Saine']).sum()}")
    print(f"  Black images: {df['is_black'].sum()}")
    print("\nTop 10 plant-disease combinations:")
    class_counts = df.groupby(['nom_plante', 'nom_maladie']).size().sort_values(ascending=False)
    print(class_counts.head(10))
    
    print("\nDone!")


if __name__ == "__main__":
    main()
