import os
import pickle
# --- PyTorch specific imports ---
import torch
import torchvision.transforms as transforms
from PIL import Image
# --- Local imports ---
from src.config import IMAGE_DATASET_PATH, FEATURES_PATH

def extract_features_pytorch(model, model_name):
    """
    Extrait les caractéristiques de toutes les images du dataset avec un modèle PyTorch.
    """
    # Définir la transformation d'image pour PyTorch
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Utiliser le GPU du Mac (MPS) si disponible
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval() # Mode évaluation

    features_list = []
    image_files = sorted([f for f in os.listdir(IMAGE_DATASET_PATH) if f.endswith('.jpg')])

    with torch.no_grad(): # Désactiver le calcul du gradient
        for img_name in image_files:
            img_path = os.path.join(IMAGE_DATASET_PATH, img_name)
            
            # Charger et pré-traiter l'image
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0) # Ajouter une dimension de batch
            image = image.to(device)

            # Extraire les caractéristiques
            feature = model(image)
            feature = feature.cpu().numpy().flatten() # Renvoyer au CPU et convertir en array numpy
            features_list.append((img_path, feature))

    # Sauvegarder les caractéristiques
    output_path = os.path.join(FEATURES_PATH, f"{model_name}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(features_list, f)
    
    print(f"Indexation pour {model_name} terminée. Fichier sauvegardé : {output_path}")
