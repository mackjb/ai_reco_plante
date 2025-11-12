# PlantVillage Dataset

## Téléchargement sans authentification

Le dataset PlantVillage est téléchargé via **Git depuis GitHub**, **sans besoin d'identifiants Kaggle** !

## Méthode de téléchargement

### Git Clone depuis GitHub

**Avantages** :
- ✅ Pas d'authentification requise
- ✅ Compatible avec les proxys SOCKS5
- ✅ 54,303 images, 38 classes
- ✅ 3 versions disponibles: `color`, `grayscale`, `segmented`

**Utilisation** :
```bash
make plantvillage
```

Le script va automatiquement :
1. Cloner le repository depuis https://github.com/spMohanty/PlantVillage-Dataset
2. Utiliser la version `color` (images RGB originales)
3. Créer un échantillon réduit (5 images/classe) pour développement rapide

## Structure du dataset

Après téléchargement :
```
dataset/plantvillage/data/
├── PlantVillage-Dataset/       # Repository GitHub
│   └── raw/
│       ├── color/              # Dataset complet RGB (54k images)
│       │   ├── Apple___Apple_scab/
│       │   ├── Apple___Black_rot/
│       │   ├── ...
│       │   └── Tomato___Yellow_Leaf_Curl_Virus/
│       ├── grayscale/          # Version niveaux de gris
│       └── segmented/          # Version segmentée
└── plantvillage_5images/       # Échantillon réduit (5 images/classe)
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    ├── ...
    └── Tomato___Yellow_Leaf_Curl_Virus/
```

## Prérequis

- **Git** : requis pour le téléchargement
  ```bash
  sudo apt-get install git
  ```

## Configuration proxy (optionnel)

Si vous êtes derrière un proxy SOCKS5 :
```bash
git config --global http.proxy socks5://localhost:1080
git config --global https.proxy socks5://localhost:1080
```

## Forcer le re-téléchargement

```bash
FORCE_DOWNLOAD=1 make plantvillage
```

## Citation

Si vous utilisez ce dataset, veuillez citer :

```bibtex
@article{hughes2015open,
  title={An open access repository of images on plant health to enable the development of mobile disease diagnostics},
  author={Hughes, David P and Salath{\'e}, Marcel},
  journal={arXiv preprint arXiv:1511.08060},
  year={2015}
}
```

## Notes

- Le dataset est hébergé sur GitHub : https://github.com/spMohanty/PlantVillage-Dataset
- Taille du clone : ~3-4 GB (shallow clone avec --depth 1)
- Compatible avec les environnements nécessitant un proxy
