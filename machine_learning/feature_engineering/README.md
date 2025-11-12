# Feature Engineering Module

Module d'extraction de features pour la reconnaissance des maladies de plantes.

## Structure

```
machine_learning/feature_engineering/
├── __init__.py                              # Package initialization
├── features.py                              # Module d'extraction de features
├── create_input_csv.py                      # Script pour créer le CSV d'entrée
├── generate_clean_with_feature_csv.py       # Script principal pour extraire les features
└── README.md                                # Cette documentation
```

## Features Extraites

Le module `features.py` implémente l'extraction de 35 features par image:

### 1. Features de Forme (5 features)
- `aire`: Surface du contour principal
- `périmètre`: Périmètre du contour principal
- `circularité`: Mesure de circularité (1.0 = cercle parfait)
- `excentricité`: Excentricité de la forme
- `aspect_ratio`: Ratio largeur/hauteur

### 2. Features de Couleur RGB (6 features)
- `mean_R`, `mean_G`, `mean_B`: Moyennes des canaux RGB
- `std_R`, `std_G`, `std_B`: Écarts-types des canaux RGB

### 3. Features HSV (3 features)
- `mean_H`, `mean_S`, `mean_V`: Moyennes des canaux HSV

### 4. Features de Texture GLCM (5 features)
- `contrast`: Contraste de texture
- `energy`: Énergie de texture
- `homogeneity`: Homogénéité
- `dissimilarité`: Dissimilarité
- `Correlation`: Corrélation

### 5. Densité de Contours (1 feature)
- `contour_density`: Ratio de pixels de contours (Canny)

### 6. Moments de Hu (7 features)
- `hu_1` à `hu_7`: Moments invariants de Hu (log-scale)

### 7. Netteté (1 feature)
- `netteté`: Variance du Laplacien

### 8. Features HOG (3 features)
- `hog_mean`: Moyenne des descripteurs HOG
- `hog_std`: Écart-type des descripteurs HOG
- `hog_entropy`: Entropie des descripteurs HOG

### 9. Features FFT (4 features)
- `fft_energy`: Énergie spectrale totale
- `fft_entropy`: Entropie du spectre
- `fft_low_freq_power`: Puissance basse fréquence
- `fft_high_freq_power`: Puissance haute fréquence

## Utilisation

### 1. Créer le CSV d'entrée

```bash
python -m machine_learning.feature_engineering.create_input_csv
```

Cela scanne le répertoire `dataset/plantvillage/data/PlantVillage-Dataset/raw/segmented` et crée:
- **Input**: Images dans `dataset/plantvillage/data/PlantVillage-Dataset/raw/segmented/`
- **Output**: `dataset/plantvillage/csv/clean_data_plantvillage_segmented_all.csv`

### 2. Extraire les features

```bash
# Sur tout le dataset
python -m machine_learning.feature_engineering.generate_clean_with_feature_csv

# Ou avec un échantillon limité (pour test)
python -m machine_learning.feature_engineering.generate_clean_with_feature_csv --limit 100
```

**Paramètres disponibles:**
- `--input`: Chemin du CSV d'entrée (défaut: `dataset/plantvillage/csv/clean_data_plantvillage_segmented_all.csv`)
- `--output`: Chemin du CSV de sortie (défaut: `dataset/plantvillage/csv/clean_data_plantvillage_segmented_all_with_features.csv`)
- `--image-col`: Nom de la colonne contenant les chemins d'images (défaut: `Image_Path`)
- `--target-size`: Taille de redimensionnement des images (défaut: `224,224`)
- `--limit`: Nombre d'images à traiter (pour test)

### 3. Utiliser les features dans votre code

```python
from machine_learning.feature_engineering import extract_all_features
import numpy as np
from PIL import Image

# Charger une image
img = Image.open("path/to/image.jpg").convert("RGB")
rgb_array = np.array(img)

# Extraire toutes les features
features = extract_all_features(rgb_array)

# features est un dictionnaire avec 35 clés
print(features.keys())
```

## Résultats

### Dataset traité
- **Total d'images**: 54,306 images
- **Classes**: 38 classes de maladies
- **Images saines**: 15,084 images
- **Images malades**: 39,222 images

### Fichier de sortie
- **Fichier**: `dataset/plantvillage/csv/clean_data_plantvillage_segmented_all_with_features.csv`
- **Taille**: 46 MB
- **Lignes**: 54,307 (54,306 images + 1 header)
- **Colonnes**: 40 (5 colonnes originales + 35 features)

### Performance
- **Vitesse de traitement**: ~40 images/seconde
- **Temps total**: ~22 minutes pour 54,306 images

## Colonnes du CSV de sortie

1. **Colonnes originales** (5):
   - `Image_Path`: Chemin complet de l'image
   - `Plante`: Nom de la plante
   - `Maladie`: Nom de la maladie (ou "healthy")
   - `Est_Saine`: Boolean (True si saine, False si malade)
   - `Label`: Label complet (format: "Plante___Maladie")

2. **Features extraites** (35): Voir section "Features Extraites" ci-dessus

## Dépendances

- numpy
- pandas
- opencv-python (cv2)
- scikit-image
- Pillow
- tqdm
