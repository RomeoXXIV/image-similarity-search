import os
import pickle
import numpy as np
# --- Local imports ---
from src.config import FEATURES_PATH, MODELS_TO_INDEX
# --- PyTorch specific imports ---
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image

# 1. FONCTIONS DE SIMILARITÉ
# ==============================================================================
def euclidean_distance(feature1, feature2):
    """Calcule la distance euclidienne entre deux vecteurs de caractéristiques."""
    return np.sqrt(np.sum((feature1 - feature2) ** 2))

def chi_square_distance(feature1, feature2):
    """Calcule la distance du chi-carré entre deux vecteurs de caractéristiques."""
    # Éviter la division par zéro
    eps = 1e-10
    return 0.5 * np.sum(((feature1 - feature2) ** 2) / (feature1 + feature2 + eps))

def correlation_distance(feature1, feature2):
    """Calcule la distance de corrélation entre deux vecteurs de caractéristiques."""
    # Normalisation des vecteurs
    feature1_norm = feature1 - np.mean(feature1)
    feature2_norm = feature2 - np.mean(feature2)
    
    # Éviter la division par zéro
    eps = 1e-10
    numerator = np.sum(feature1_norm * feature2_norm)
    denominator = np.sqrt(np.sum(feature1_norm**2) * np.sum(feature2_norm**2)) + eps
    
    # La corrélation est entre -1 et 1, où 1 signifie parfaitement corrélé
    # Nous voulons une distance, donc nous utilisons 1 - corrélation
    correlation = numerator / denominator
    return 1.0 - correlation

def bhattacharyya_distance(feature1, feature2):
    """Calcule la distance de Bhattacharyya entre deux vecteurs de caractéristiques."""
    # Normalisation pour s'assurer que les vecteurs sont non-négatifs et somment à 1
    eps = 1e-10
    feature1_norm = feature1 / (np.sum(feature1) + eps)
    feature2_norm = feature2 / (np.sum(feature2) + eps)
    
    # Calcul de la distance de Bhattacharyya
    bc = np.sum(np.sqrt(feature1_norm * feature2_norm))
    return -np.log(bc + eps)

def cosine_distance(feature1, feature2):
    """Calcule la distance cosinus entre deux vecteurs de caractéristiques."""
    # Éviter la division par zéro
    eps = 1e-10
    dot_product = np.sum(feature1 * feature2)
    norm_product = np.sqrt(np.sum(feature1**2)) * np.sqrt(np.sum(feature2**2)) + eps
    similarity = dot_product / norm_product
    
    # La similarité cosinus est entre -1 et 1, où 1 signifie parfaitement similaire
    # Nous voulons une distance, donc nous utilisons 1 - similarité
    return 1.0 - similarity

# 2. CHARGEMENT DES DONNÉES
# ==============================================================================
def load_features():
    """Charge tous les fichiers de caractéristiques .pkl du dossier."""
    features_data = {}
    for model_name in MODELS_TO_INDEX.keys():
        path = os.path.join(FEATURES_PATH, f"{model_name}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                features_data[model_name] = pickle.load(f)
            print(f"Descripteurs pour '{model_name}' chargés.")
        else:
            print(f"Attention : Fichier de descripteurs introuvable pour '{model_name}' à l'emplacement {path}")
    return features_data

# 3. EXTRACTION DE CARACTÉRISTIQUES POUR UNE IMAGE REQUÊTE
# ==============================================================================
def extract_query_features(image_path, model_name):
    """
    Extrait les caractéristiques d'une seule image requête avec un modèle PyTorch.
    """
    # 1. Charger le bon modèle
    if model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = torch.nn.Identity()
    else:
        raise ValueError("Modèle non supporté.")

    # 2. Appliquer les transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3. Extraire et retourner les features
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        feature = model(image)
        return feature.cpu().numpy().flatten()


# 4. FONCTION DE RECHERCHE PRINCIPALE
# ==============================================================================
def search(query_path, model_name, all_features, distance_metric='euclidean', top_n=10):
    """
    Recherche les images les plus similaires à l'image requête.
    
    Args:
        query_path (str): Chemin vers l'image requête
        model_name (str): Nom du modèle à utiliser
        all_features (dict): Dictionnaire contenant les caractéristiques extraites
        distance_metric (str): Métrique de distance à utiliser ('euclidean', 'chi_square', etc.)
        top_n (int): Nombre de résultats à retourner
        
    Returns:
        list: Liste des chemins des images les plus similaires avec leur score
    """
    # Extraire les caractéristiques de l'image requête
    query_features = extract_query_features(query_path, model_name)
    
    # Sélectionner la fonction de distance appropriée
    if distance_metric == 'euclidean':
        distance_fn = euclidean_distance
    elif distance_metric == 'chi_square':
        distance_fn = chi_square_distance
    elif distance_metric == 'correlation':
        distance_fn = correlation_distance
    elif distance_metric == 'bhattacharyya':
        distance_fn = bhattacharyya_distance
    elif distance_metric == 'cosine':
        distance_fn = cosine_distance
    else:
        print(f"Métrique de distance '{distance_metric}' non reconnue. Utilisation de la distance euclidienne par défaut.")
        distance_fn = euclidean_distance

    # Récupérer les caractéristiques de la base de données pour le modèle choisi
    dataset_features = all_features.get(model_name)
    if not dataset_features:
        raise ValueError(f"Aucun descripteur chargé pour le modèle '{model_name}'.")

    # Calculer les distances
    results = []
    for img_path, feature_vector in dataset_features:
        dist = distance_fn(query_features, feature_vector)
        results.append((img_path, dist))

    # Trier par distance et retourner le top N
    results.sort(key=lambda x: x[1])
    return results[:top_n]
