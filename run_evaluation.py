import numpy as np
import os
import time
from tqdm import tqdm # Une belle barre de progression
import matplotlib.pyplot as plt

# --- Local imports ---
from src.config import MODELS_TO_INDEX, IMAGE_DATASET_PATH
from src.retrieval import load_features, search
from src.evaluation import get_image_class, calculate_average_precision

def main():
    """
    Script principal pour évaluer et comparer les modèles en utilisant la MAP et tracer les courbes R/P.
    """
    print("--- Lancement de l'évaluation des modèles ---")

    # 1. Charger tous les descripteurs en mémoire
    print("Chargement des descripteurs...")
    all_features = load_features()
    print("Descripteurs chargés.")

    # 2. Définir les images requêtes pour l'évaluation
    # Pour un bon score, on prend plusieurs images de chaque classe.
    # On prend 10 images par classe pour une meilleure représentativité
    query_image_paths = []
    for class_start in range(0, 1000, 100):  # Pour chaque classe (0, 100, 200, ...)
        for offset in range(0, 100, 10):      # Prendre 5 images par classe (0, 10, 20, 30, 40)
            img_path = os.path.join(IMAGE_DATASET_PATH, f"{class_start + offset}.jpg")
            if os.path.exists(img_path):
                query_image_paths.append(img_path)
    
    print(f"Évaluation sur {len(query_image_paths)} images requêtes")
    
    # 3. Boucler sur chaque modèle pour l'évaluer
    final_results = {}
    # Dictionnaire pour stocker les points des courbes pour chaque requête
    all_curves = {}
    
    # Définir les méthodes de similarité à évaluer
    similarity_metrics = ['euclidean', 'chi_square', 'correlation', 'bhattacharyya', 'cosine']
    
    # Créer le dossier 'results' s'il n'existe pas
    if not os.path.exists('results'):
        os.makedirs('results')

    for model_name in MODELS_TO_INDEX.keys():
        print(f"\n--- Évaluation du modèle : {model_name} ---")
        
        # Stocker les résultats pour chaque métrique de similarité
        for similarity in similarity_metrics:
            model_similarity_key = f"{model_name}_{similarity}"
            print(f"  Utilisation de la métrique : {similarity}")
            
            average_precisions = []
            start_time = time.time()
            
            # Stocker les courbes pour chaque requête
            query_curves = {}
            
            # Utiliser tqdm pour une barre de progression
            for i, query_path in enumerate(tqdm(query_image_paths, desc=f"Requêtes pour {model_name} ({similarity})")):
                # Lancer une recherche pour la requête sur TOUTE la base de données
                ranked_results = search(query_path, model_name, all_features, distance_metric=similarity, top_n=1000)

                # Obtenir la vérité terrain
                query_class = get_image_class(query_path)
                total_relevant_docs = 100 # Il y a 100 images dans chaque classe

                # Calculer l'AP pour cette requête ET récupérer les points de la courbe
                ap, pr_points = calculate_average_precision(ranked_results, query_class, total_relevant_docs)
                average_precisions.append(ap)
                
                # Sauvegarder la courbe pour cette requête
                query_id = os.path.basename(query_path).split('.')[0]
                query_curves[query_id] = pr_points
            
            # Calculer la MAP (Mean Average Precision)
            map_score = np.mean(average_precisions)
            end_time = time.time()

            final_results[model_similarity_key] = {
                'map': map_score,
                'duration': end_time - start_time
            }
            
            all_curves[model_similarity_key] = query_curves
            
            # Générer une courbe pour chaque requête
            for query_id, pr_points in query_curves.items():
                plt.figure(figsize=(10, 8))
                recalls, precisions = pr_points
                
                if recalls and precisions:
                    plt.plot(recalls, precisions, marker='o', linestyle='--', 
                             label=f'{model_name.upper()} - {similarity}')
                    
                    plt.xlabel('Rappel (Recall)', fontsize=14)
                    plt.ylabel('Précision (Precision)', fontsize=14)
                    plt.title(f'Courbe Rappel/Précision pour la requête "{query_id}.jpg"', fontsize=16)
                    plt.legend(fontsize=12)
                    plt.grid(True)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    
                    output_path = f'results/pr_curve_{model_name}_{similarity}_{query_id}.png'
                    plt.savefig(output_path)
                    plt.close()

    # 4. Afficher les résultats finaux
    print("\n\n--- Résultats finaux de l'évaluation ---")
    print("-----------------------------------------")
    for result_key, scores in final_results.items():
        # Correction pour gérer les clés avec plusieurs underscores
        parts = result_key.split('_')
        model = parts[0]
        similarity = '_'.join(parts[1:])  # Combine les parties restantes pour la similarité
        print(f"Modèle : {model.upper()} - Similarité : {similarity}")
        print(f"  - Score MAP      : {scores['map']:.4f}")
        print(f"  - Durée totale   : {scores['duration']:.2f} secondes")
        print("-----------------------------------------")
    
    best_result = max(final_results, key=lambda m: final_results[m]['map'])
    # Correction pour gérer les clés avec plusieurs underscores
    parts = best_result.split('_')
    best_model = parts[0]
    best_similarity = '_'.join(parts[1:])
    print(f"\nLa meilleure combinaison est : {best_model.upper()} avec {best_similarity} - Score MAP de {final_results[best_result]['map']:.4f}")

    # 5. Générer un graphique comparatif des courbes Rappel/Précision moyennes
    print("\nGénération du graphique comparatif des courbes Rappel/Précision moyennes...")
    plt.figure(figsize=(14, 10))  # Agrandir légèrement le graphique
    
    # Palette de couleurs plus distinctes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#1a55FF', '#FF5733', '#33FF57', '#FF33A8', '#33A8FF']
    
    # Styles de ligne plus visibles
    styles = ['-', '--', '-.', ':']
    
    # Marqueurs variés
    markers = ['o', 's', '^', 'D', '*', 'x', '+']
    
    color_idx = 0
    for result_key, query_curves in all_curves.items():
        # Correction pour gérer les clés avec plusieurs underscores
        parts = result_key.split('_')
        model = parts[0]
        similarity = '_'.join(parts[1:])
        
        # Calculer les courbes moyennes pour toutes les requêtes
        avg_recalls = []
        avg_precisions = []
        
        # Collecter tous les points de rappel uniques à travers toutes les requêtes
        all_recall_points = set()
        for query_id, (recalls, _) in query_curves.items():
            all_recall_points.update(recalls)
        
        all_recall_points = sorted(list(all_recall_points))
        
        # Pour chaque point de rappel, calculer la précision moyenne
        for recall_point in all_recall_points:
            precisions_at_recall = []
            
            for query_id, (recalls, precisions) in query_curves.items():
                # Trouver l'index du point de rappel le plus proche
                if recalls:  # Vérifier que la liste n'est pas vide
                    closest_idx = min(range(len(recalls)), key=lambda i: abs(recalls[i] - recall_point))
                    # Si le rappel est suffisamment proche, ajouter la précision correspondante
                    if abs(recalls[closest_idx] - recall_point) < 0.05:  # Tolérance de 0.05
                        precisions_at_recall.append(precisions[closest_idx])
            
            # S'il y a des précisions à ce point de rappel, calculer la moyenne
            if precisions_at_recall:
                avg_recalls.append(recall_point)
                avg_precisions.append(sum(precisions_at_recall) / len(precisions_at_recall))
        
        # Tracer la courbe moyenne si nous avons des points
        if avg_recalls and avg_precisions:
            # Utiliser une combinaison unique de couleur, style et marqueur
            plt.plot(avg_recalls, avg_precisions, 
                     color=colors[color_idx % len(colors)],
                     linestyle=styles[(color_idx // len(colors)) % len(styles)],
                     marker=markers[(color_idx // (len(colors) * len(styles))) % len(markers)],
                     markersize=4,  # Réduire la taille des marqueurs
                     linewidth=2,   # Augmenter l'épaisseur des lignes
                     label=f'{model.upper()} - {similarity}')
            color_idx += 1
    
    plt.xlabel('Rappel (Recall)', fontsize=14)
    plt.ylabel('Précision (Precision)', fontsize=14)
    plt.title(f'Comparaison des courbes Rappel/Précision moyennes', fontsize=16)
    plt.legend(fontsize=12, loc='lower left', bbox_to_anchor=(0, 0), ncol=2)  # Légende plus lisible
    plt.grid(True, alpha=0.3)  # Grille plus discrète
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    output_path = 'results/comparison_pr_curves_mean.png'
    plt.savefig(output_path, dpi=300)  # Augmenter la résolution
    print(f"Graphique comparatif sauvegardé dans : {output_path}")


if __name__ == '__main__':
    main()