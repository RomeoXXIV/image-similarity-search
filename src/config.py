import os

# Chemin vers le dossier racine de la base d'images
IMAGE_DATASET_PATH = os.path.join('app', 'static', 'image.orig')

# Chemin où les fichiers de caractéristiques (.pkl) seront sauvegardés
FEATURES_PATH = os.path.join('app', 'static', 'features')

# S'assurer que le dossier de sauvegarde existe
os.makedirs(FEATURES_PATH, exist_ok=True)

# Liste des modèles que nous allons utiliser. Le nom correspondra au fichier .pkl
# Exemple : 'vgg16' -> 'vgg16.pkl'
MODELS_TO_INDEX = {
    'vgg16': 'pytorch',
    'resnet50': 'pytorch',
    'vit_b_16': 'pytorch' # Vision Transformer
}
