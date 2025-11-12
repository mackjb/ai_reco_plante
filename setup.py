"""
Setup pour le projet ai_reco_plante.
"""
from setuptools import setup, find_packages

setup(
    name="ai_reco_plante",
    version="0.1.0",
    description="Reconnaissance et classification de maladies des plantes",
    author="Your Name",
    author_email="you@example.com",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "kagglehub",
        "jupyter",
        "ipykernel",
        "pandas",
        "numpy",
        "plotly",
        "Pillow",
        "matplotlib",
        "opencv-python-headless",
        "scikit-learn",
        "albumentations",
        "scikit-image",
        "imbalanced-learn",
        "seaborn",
        "torchvision",
        "shap",
        "joblib",
    ],
)
