import torch
import torchvision.models as models
# --- Local imports ---
from src.indexing import extract_features_pytorch

def main():
    print("Début du processus d'indexation...")

    # --- Modèle 1: VGG16 ---
    model_vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # On enlève la dernière couche (le classifieur) pour obtenir le vecteur de features
    model_vgg16.classifier = torch.nn.Sequential(*list(model_vgg16.classifier.children())[:-1])
    extract_features_pytorch(model_vgg16, 'vgg16')

    # --- Modèle 2: ResNet50 ---
    model_resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # On enlève la dernière couche (fully connected)
    model_resnet50.fc = torch.nn.Identity()
    extract_features_pytorch(model_resnet50, 'resnet50')

    # --- Modèle 3: Vision Transformer ---
    model_vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    # On enlève la tête de classification
    model_vit.heads.head = torch.nn.Identity()
    extract_features_pytorch(model_vit, 'vit_b_16')

    print("\nTous les modèles ont été indexés avec succès.")

if __name__ == '__main__':
    # Avant de lancer, supprime les anciens fichiers .pkl pour être propre
    import os, glob
    from src.config import FEATURES_PATH
    old_files = glob.glob(os.path.join(FEATURES_PATH, '*.pkl'))
    for f in old_files:
        os.remove(f)
        print(f"Ancien fichier {os.path.basename(f)} supprimé.")

    main()

