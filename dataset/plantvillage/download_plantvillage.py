import os
import shutil
import sys
import subprocess
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))
from helpers import PROJECT_ROOT




def duplicate_dataset_limited(src_dir, dst_dir, max_files_per_class=5):
    """
    Copie la structure de dossiers de src_dir vers dst_dir en ne gardant que max_files_per_class fichiers image par sous-dossier.
    
    Args:
        src_dir (str): chemin vers dataset source
        dst_dir (str): chemin vers dataset destination
        max_files_per_class (int): nombre max d'images à copier par sous-dossier
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(src_dir):
        # Calcul chemin relatif depuis src_dir
        rel_path = os.path.relpath(root, src_dir)
        # Nouveau chemin dans dst_dir
        target_dir = os.path.join(dst_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        
        # Filtrer uniquement fichiers images jpg/jpeg/png (en minuscules)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files = sorted(image_files)[:max_files_per_class]  # Prendre les 5 premières
        
        for file in image_files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_dir, file)
            shutil.copy2(src_file, dst_file)  # copie avec métadonnées

    print(f"Copie terminée dans {dst_dir} (max {max_files_per_class} images par dossier)")



def download_plantvillage_dataset(dst_dir: Path) -> Path:
    """
    Clone le dataset PlantVillage depuis GitHub.
    Retourne le chemin du dataset si succès, sinon lève une exception.
    """
    if shutil.which("git") is None:
        raise RuntimeError("❌ Git n'est pas installé. Installez-le avec: sudo apt-get install git")
    
    repo_url = "https://github.com/spMohanty/PlantVillage-Dataset.git"
    clone_dir = dst_dir / "PlantVillage-Dataset"
    
    print(f"⬇️  Téléchargement depuis GitHub...")
    print(f"   URL: {repo_url}")
    print(f"   Destination: {clone_dir}")
    
    # Cloner le repo (shallow clone pour économiser de la bande passante)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--progress", repo_url, str(clone_dir)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ Échec du clone Git: {e}")
    
    print(f"✅ Repository cloné avec succès!")
    
    # Le dataset est dans le sous-dossier raw/color
    src_dataset = clone_dir / "raw" / "color"
    if not src_dataset.exists():
        raise RuntimeError(f"⚠️  Structure inattendue: {src_dataset} introuvable")
    
    print(f"📂 Images trouvées: {src_dataset}")
    return src_dataset



if __name__ == "__main__":
    project_root = PROJECT_ROOT
    dst = project_root / "dataset" / "plantvillage" / "data"
    dst_dataset = dst / "plantvillage_5images"
    force = os.environ.get("FORCE_DOWNLOAD", "").lower() in ("1", "true", "yes", "on")

    print("=" * 60)
    print("🌱 PlantVillage Dataset - Téléchargement")
    print("=" * 60)

    # Idempotence: si déjà prêt, sortir
    if dst_dataset.exists() and not force:
        print(f"✅ Le dataset existe déjà à : {dst_dataset}")
        print("💡 Utilisez FORCE_DOWNLOAD=1 pour forcer le téléchargement")
        sys.exit(0)

    # Si on force, on nettoie la destination pour éviter les collisions
    if force and dst.exists():
        print(f"♻️  FORCE_DOWNLOAD actif: suppression de {dst}")
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)

    # Télécharger depuis GitHub
    try:
        src_dataset = download_plantvillage_dataset(dst)
    except RuntimeError as e:
        print(f"\n{e}")
        print("\n💡 Solutions possibles:")
        print("1. Vérifier votre connexion Internet / proxy")
        print("2. Installer git: sudo apt-get install git")
        print("3. Si vous êtes derrière un proxy, configurez Git:")
        print("   git config --global http.proxy <proxy_url>")
        sys.exit(1)

    # Créer un échantillon réduit (5 images par classe)
    print(f"\n📂 Création d'un échantillon réduit...")
    has_subdirs = any((src_dataset / d).is_dir() for d in os.listdir(src_dataset))
    
    if has_subdirs:
        duplicate_dataset_limited(src_dataset, dst_dataset, max_files_per_class=5)
        print(f"\n✅ Dataset PlantVillage prêt !")
        print(f"   📁 Dataset complet: {src_dataset}")
        print(f"   📁 Échantillon (5/classe): {dst_dataset}")
        print(f"\n🎯 Utilisez {dst_dataset} pour le développement rapide")
    else:
        print(f"⚠️  Structure du dataset inattendue dans {src_dataset}")
        sys.exit(1)

