#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a cleaned CSV with image features for PlantVillage segmented dataset.

Input CSV (default): dataset/plantvillage/csv/clean_data_plantvillage_segmented_all.csv
Output CSV (default): dataset/plantvillage/csv/clean_data_plantvillage_segmented_all_with_features.csv

Implements the following features per image:
- Shape: aire, périmètre, circularité, excentricité, aspect_ratio
- Color (RGB): mean_R, mean_G, mean_B, std_R, std_G, std_B
- HSV means: mean_H, mean_S, mean_V
- Texture (GLCM): contrast, energy, homogeneity, dissimilarité, Correlation
- Contour density: contour_density
- Hu moments: hu_1..hu_7 (log-scaled)
- Sharpness: netteté (variance of Laplacian)
- HOG: hog_mean, hog_std, hog_entropy
- FFT: fft_energy, fft_entropy, fft_low_freq_power, fft_high_freq_power

Separation of concerns: each feature family has its own function, orchestrated by extract_all_features.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

# Import feature extraction functions from local module
from .features import extract_all_features


# ------------------------- I/O and pipeline -------------------------

def load_image_rgb(path: str, target_size: Optional[tuple[int, int]] = (224, 224)) -> Optional[np.ndarray]:
    """
    Load an image as RGB uint8. Optionally resize.
    Returns None if the image cannot be loaded.
    """
    try:
        img = Image.open(path).convert("RGB")
        if target_size is not None:
            img = img.resize(target_size)
        return np.array(img)
    except Exception:
        return None


def process_dataframe(
    df: pd.DataFrame,
    image_col: str = "Image_Path",
    target_size: Optional[tuple[int, int]] = (224, 224),
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Iterate rows, compute features, and return a concatenated DataFrame with original columns + features.
    Column order matches the old CSV format:
    - Metadata: nom_plante, nom_maladie, Est_Saine, Image_Path, width_img, height_img, is_black, md5
    - Features: 35 feature columns
    """
    records: List[Dict[str, Any]] = []

    it = df.itertuples(index=False)
    if limit is not None:
        it = list(it)[:limit]

    for row in tqdm(it, total=(limit if limit is not None else len(df)), desc="Extracting features"):
        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(df.columns, row))
        img_path = row_dict.get(image_col)
        rgb = load_image_rgb(img_path, target_size=target_size)
        if rgb is None:
            # keep row but with NaN features
            feats = {k: np.nan for k in [
                "aire", "périmètre", "circularité", "excentricité", "aspect_ratio",
                "mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B",
                "mean_H", "mean_S", "mean_V",
                "contrast", "energy", "homogeneity", "dissimilarité", "Correlation",
                "contour_density",
                "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6", "hu_7",
                "netteté",
                "hog_mean", "hog_std", "hog_entropy",
                "fft_energy", "fft_entropy", "fft_low_freq_power", "fft_high_freq_power",
            ]}
        else:
            feats = extract_all_features(rgb)
        records.append({**row_dict, **feats})

    result_df = pd.DataFrame.from_records(records)
    
    # Ensure column order matches old CSV format
    metadata_cols = ['nom_plante', 'nom_maladie', 'Est_Saine', 'Image_Path', 
                     'width_img', 'height_img', 'is_black', 'md5']
    feature_cols = [
        'aire', 'périmètre', 'circularité', 'excentricité', 'aspect_ratio',
        'mean_R', 'mean_G', 'mean_B', 'std_R', 'std_G', 'std_B',
        'mean_H', 'mean_S', 'mean_V',
        'contrast', 'energy', 'homogeneity', 'dissimilarité', 'Correlation',
        'contour_density',
        'hu_1', 'hu_2', 'hu_3', 'hu_4', 'hu_5', 'hu_6', 'hu_7',
        'netteté',
        'hog_mean', 'hog_std', 'hog_entropy',
        'fft_energy', 'fft_entropy', 'fft_low_freq_power', 'fft_high_freq_power'
    ]
    
    # Keep only columns that exist
    final_cols = [col for col in metadata_cols if col in result_df.columns] + feature_cols
    return result_df[final_cols]


def main():
    parser = argparse.ArgumentParser(description="Generate CSV with image features for PlantVillage segmented dataset")
    default_input = str(Path("dataset/plantvillage/csv/clean_data_plantvillage_segmented_all.csv"))
    default_output = str(Path("dataset/plantvillage/csv/clean_data_plantvillage_segmented_all_with_features.csv"))
    parser.add_argument("--input", "--input_csv", dest="input_csv", default=default_input, help="Path to input CSV")
    parser.add_argument("--output", "--output_csv", dest="output_csv", default=default_output, help="Path to output CSV")
    parser.add_argument("--image-col", dest="image_col", default="Image_Path", help="Column name with image paths")
    parser.add_argument("--target-size", dest="target_size", default="224,224", help="Resize images to WxH, or 'none'")
    parser.add_argument("--limit", dest="limit", type=int, default=None, help="Process only first N rows (debug)")

    args = parser.parse_args()

    if args.target_size and isinstance(args.target_size, str) and args.target_size.lower() != "none":
        try:
            w, h = map(int, args.target_size.split(','))
            target_size = (w, h)
        except Exception:
            raise ValueError("--target-size must be 'none' or 'W,H'")
    else:
        target_size = None

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)

    # Type normalization for expected columns if present
    for bcol in ["Est_Saine", "is_black"]:
        if bcol in df.columns:
            # Convert strings 'True'/'False' to int 1/0
            df[bcol] = df[bcol].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(df[bcol]).astype(str)
            df[bcol] = df[bcol].map({'1': 1, '0': 0}).fillna(0).astype(int)

    df_features = process_dataframe(df, image_col=args.image_col, target_size=target_size, limit=args.limit)

    print(f"Writing output CSV: {output_csv}")
    df_features.to_csv(output_csv, index=False, encoding='utf-8')
    print("Done.")


if __name__ == "__main__":
    main()
