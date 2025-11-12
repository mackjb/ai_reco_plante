#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature extraction functions for plant disease recognition.

Implements comprehensive feature extractors:
- Shape features: area, perimeter, circularity, eccentricity, aspect_ratio
- Color features (RGB): mean and std per channel
- HSV features: mean HSV components
- Texture features (GLCM): contrast, energy, homogeneity, dissimilarity, correlation
- Contour density: edge pixel ratio
- Hu moments: 7 log-scaled invariant moments
- Sharpness: variance of Laplacian
- HOG features: mean, std, entropy
- FFT features: energy, entropy, low/high frequency power
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import cv2
import numpy as np

# skimage imports with fallback for different versions
try:
    from skimage.feature import graycomatrix, graycoprops, hog  # type: ignore
except Exception:  # pragma: no cover
    try:
        from skimage.feature.texture import graycomatrix, graycoprops  # type: ignore
        from skimage.feature import hog  # type: ignore
    except Exception as e:  # pragma: no cover
        raise e

try:
    from skimage.measure import label, regionprops
except Exception:  # pragma: no cover
    label = None
    regionprops = None


def extract_shape_features(gray_img: np.ndarray, binary_thresh: Optional[int] = None) -> Dict[str, float]:
    """
    Shape features from the largest contour.
    
    Args:
        gray_img: Grayscale image (2D numpy array)
        binary_thresh: Optional threshold for binarization (uses Otsu if None)
    
    Returns:
        Dictionary with: aire, périmètre, circularité, excentricité, aspect_ratio
    """
    if gray_img.ndim != 2:
        raise ValueError("extract_shape_features expects a grayscale image")

    # Binarization: Otsu if threshold not provided
    if binary_thresh is None:
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray_img, binary_thresh, 255, cv2.THRESH_BINARY)

    # Find external contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Return NaNs to signal missing contour
        return {
            "aire": np.nan,
            "périmètre": np.nan,
            "circularité": np.nan,
            "excentricité": np.nan,
            "aspect_ratio": np.nan,
        }

    cnt = max(contours, key=cv2.contourArea)

    aire = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    circularite = (4.0 * math.pi * aire) / (perim ** 2) if perim > 0 else 0.0

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0

    # Eccentricity via regionprops on labeled mask if available
    excentricite = 0.0
    if label is not None and regionprops is not None:
        lbl = label(binary > 0)
        props = regionprops(lbl)
        if props:
            largest_region = max(props, key=lambda p: p.area)
            excentricite = float(largest_region.eccentricity)

    return {
        "aire": aire,
        "périmètre": perim,
        "circularité": circularite,
        "excentricité": excentricite,
        "aspect_ratio": float(aspect_ratio),
    }


def extract_color_features(rgb_img: np.ndarray) -> Dict[str, float]:
    """
    Color features on RGB channels: mean and std per channel.
    
    Args:
        rgb_img: RGB uint8 image (H x W x 3)
    
    Returns:
        Dictionary with: mean_R, mean_G, mean_B, std_R, std_G, std_B
    """
    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError("extract_color_features expects an RGB image")

    R, G, B = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
    return {
        "mean_R": float(np.mean(R)),
        "mean_G": float(np.mean(G)),
        "mean_B": float(np.mean(B)),
        "std_R": float(np.std(R)),
        "std_G": float(np.std(G)),
        "std_B": float(np.std(B)),
    }


def extract_hsv_features(rgb_img: np.ndarray) -> Dict[str, float]:
    """
    Mean HSV components.
    
    Args:
        rgb_img: RGB uint8 image (H x W x 3)
    
    Returns:
        Dictionary with: mean_H, mean_S, mean_V
    """
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    return {
        "mean_H": float(np.mean(h)),
        "mean_S": float(np.mean(s)),
        "mean_V": float(np.mean(v)),
    }


def extract_texture_features(gray_img: np.ndarray, levels: int = 256) -> Dict[str, float]:
    """
    Texture features via GLCM for distance=1, angle=0.
    
    Args:
        gray_img: Grayscale image (2D numpy array)
        levels: Number of gray levels for GLCM computation
    
    Returns:
        Dictionary with: contrast, energy, homogeneity, dissimilarité, Correlation
    """
    if gray_img.ndim != 2:
        raise ValueError("extract_texture_features expects a grayscale image")
    
    # skimage expects values in [0, levels-1]
    if levels != 256:
        # Quantize if different levels requested
        gray = np.floor(gray_img.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    else:
        gray = gray_img

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)

    contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    energy = float(graycoprops(glcm, 'energy')[0, 0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    dissimilarite = float(graycoprops(glcm, 'dissimilarity')[0, 0])
    correlation = float(graycoprops(glcm, 'correlation')[0, 0])

    return {
        "contrast": contrast,
        "energy": energy,
        "homogeneity": homogeneity,
        "dissimilarité": dissimilarite,
        "Correlation": correlation,
    }


def extract_contour_density(gray_img: np.ndarray) -> Dict[str, float]:
    """
    Canny edge pixel ratio to total pixels.
    
    Args:
        gray_img: Grayscale image (2D numpy array)
    
    Returns:
        Dictionary with: contour_density
    """
    edges = cv2.Canny(gray_img, 100, 200)
    density = float(np.sum(edges > 0) / edges.size)
    return {"contour_density": density}


def extract_hu_moments(gray_img: np.ndarray) -> Dict[str, float]:
    """
    Log-scaled Hu moments (7 values).
    
    Args:
        gray_img: Grayscale image (2D numpy array)
    
    Returns:
        Dictionary with: hu_1 to hu_7
    """
    m = cv2.moments(gray_img)
    hu = cv2.HuMoments(m).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return {f"hu_{i+1}": float(hu_log[i]) for i in range(7)}


def extract_sharpness(gray_img: np.ndarray) -> Dict[str, float]:
    """
    Variance of Laplacian as a sharpness measure.
    
    Args:
        gray_img: Grayscale image (2D numpy array)
    
    Returns:
        Dictionary with: netteté
    """
    lap = cv2.Laplacian(gray_img, cv2.CV_64F)
    return {"netteté": float(np.var(lap))}


def extract_hog_features(gray_img: np.ndarray) -> Dict[str, float]:
    """
    HOG features reduced to summary stats: mean, std, entropy over descriptor.
    
    Args:
        gray_img: Grayscale image (2D numpy array)
    
    Returns:
        Dictionary with: hog_mean, hog_std, hog_entropy
    """
    vec = hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True,
    )
    vec = vec.astype(np.float64)
    mean = float(np.mean(vec))
    std = float(np.std(vec))
    p = np.abs(vec)
    s = p.sum()
    if s <= 0:
        entropy = 0.0
    else:
        p = p / s
        entropy = float(-(p * (np.log2(p + 1e-12))).sum())
    return {"hog_mean": mean, "hog_std": std, "hog_entropy": entropy}


def extract_fft_features(gray_img: np.ndarray) -> Dict[str, float]:
    """
    2D FFT magnitude power spectrum summary statistics.
    
    Args:
        gray_img: Grayscale image (2D numpy array)
    
    Returns:
        Dictionary with: fft_energy, fft_entropy, fft_low_freq_power, fft_high_freq_power
    """
    f = np.fft.fft2(gray_img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift) ** 2

    energy = float(power.sum())
    psum = power.sum()
    if psum <= 0:
        entropy = 0.0
    else:
        p = power / psum
        entropy = float(-(p * (np.log2(p + 1e-12))).sum())

    # Radial split: low vs high frequency by radius threshold
    h, w = gray_img.shape[:2]
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    r_max = np.max(r)
    r_thresh = 0.25 * r_max  # 25% of max radius considered low-frequency
    low_mask = r <= r_thresh
    high_mask = ~low_mask

    low_power = float(power[low_mask].sum())
    high_power = float(power[high_mask].sum())

    return {
        "fft_energy": energy,
        "fft_entropy": entropy,
        "fft_low_freq_power": low_power,
        "fft_high_freq_power": high_power,
    }


def extract_all_features(rgb_img: np.ndarray) -> Dict[str, float]:
    """
    Orchestrates extraction over all feature families.
    
    Args:
        rgb_img: RGB image (uint8, H x W x 3)
    
    Returns:
        Dictionary with all extracted features
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    feats: Dict[str, float] = {}
    feats.update(extract_shape_features(gray))
    feats.update(extract_color_features(rgb_img))
    feats.update(extract_hsv_features(rgb_img))
    feats.update(extract_texture_features(gray))
    feats.update(extract_contour_density(gray))
    feats.update(extract_hu_moments(gray))
    feats.update(extract_sharpness(gray))
    feats.update(extract_hog_features(gray))
    feats.update(extract_fft_features(gray))
    return feats
