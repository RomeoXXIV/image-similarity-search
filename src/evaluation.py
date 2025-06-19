import numpy as np
import os

def get_image_class(image_path):
    """
    Détermine la classe d'une image à partir de son nom de fichier.
    Exemple: 'data/image.orig/150.jpg' -> classe 1
    
    Cette fonction est conçue pour les images de la collection originale.
    Pour les images téléchargées par l'utilisateur, elle retourne None.
    """
    # Vérifier si l'image provient du dossier d'upload
    if 'uploads' in image_path:
        return None
        
    basename = os.path.basename(image_path)
    
    # Format standard pour les images de la collection: numéro.extension
    # Le numéro divisé par 100 donne la classe
    if '.' in basename and not '_' in basename:
        name_part = basename.split('.')[0]
        if name_part.isdigit():
            image_id = int(name_part)
            return image_id // 100
    
    # Si ce n'est pas au format standard, on ne peut pas déterminer la classe
    return None


def calculate_average_precision(ranked_list, query_class, total_relevant_docs):
    """
    Calcule l'Average Precision (AP) et retourne les points (recall, precision) pour la courbe.
    """
    hits = 0
    sum_precisions = 0
    
    # Points pour la courbe, on commence à (R=0, P=1) pour un beau tracé
    recall_points = [0.0]
    precision_points = [1.0]

    if total_relevant_docs == 0:
        return 0.0, (recall_points, precision_points)

    for i, (result_path, distance) in enumerate(ranked_list):
        rank = i + 1
        # On vérifie si le document est pertinent
        result_class = None
        
        # Extraire la classe à partir du nom de fichier pour les images de la base de données
        basename = os.path.basename(result_path)
        if '.' in basename and basename.split('.')[0].isdigit():
            result_class = int(basename.split('.')[0]) // 100
        
        if result_class == query_class:
            hits += 1
            precision = hits / rank
            sum_precisions += precision
            
            recall = hits / total_relevant_docs
            recall_points.append(recall)
            precision_points.append(precision)

    if hits == 0:
        return 0.0, ([], [])

    average_precision = sum_precisions / hits
    
    return average_precision, (recall_points, precision_points)
