"""
Module d'utilitaires globaux pour le projet.

Fonctionnalités:
- Gestion de la racine du projet via get_project_root()
- Validation d'images (is_image_valid, is_black_image)

Note: Les fonctions d'extraction de features ont été déplacées vers:
      machine_learning/feature_engineering/features.py
"""
from pathlib import Path
from PIL import Image, ImageStat


def get_project_root(marker: str = "setup.py") -> Path:
    """
    Retourne le Path du répertoire racine du projet (contenant le fichier `marker`).

    Args:
        marker: nom du fichier ou dossier marqueur (par défaut "setup.py").

    Returns:
        Path vers la racine du projet.

    Raises:
        FileNotFoundError: si aucun parent ne contient `marker`.
    """
    current = Path(__file__).resolve()
    for parent in (current, *current.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Impossible de trouver la racine du projet (marker={marker})")

# Instance globale accessible partout
PROJECT_ROOT: Path = get_project_root()


def is_image_valid(image_path: str) -> bool:
    """Vérifie que l'image n'est pas corrompue."""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Vérifie que l'image peut être lue correctement
        return True
    except Exception:
        return False


def is_black_image(image_path, threshold=10) -> bool:
    """
    Returns True if the image is mostly black based on the mean grayscale value.

    Args:
        image_path: Path to the image file
        threshold: Mean grayscale threshold below which image is considered black

    Returns:
        True if image is mostly black, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            stat = ImageStat.Stat(img.convert('L'))
            return stat.mean[0] < threshold
    except Exception:
        return False
