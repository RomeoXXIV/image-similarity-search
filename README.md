# Système de Recherche d'Images par Similarité

Ce projet implémente un système de recherche d'images par similarité visuelle basé sur des caractéristiques extraites par des modèles de deep learning. L'application web permet aux utilisateurs de télécharger une image et de trouver les images les plus similaires dans une base de données, en utilisant différentes métriques de similarité.

## Fonctionnalités

- **Recherche d'images par similarité** : Téléchargez une image et trouvez les images les plus similaires dans la base de données
- **Modèles multiples** : Support pour différents modèles de deep learning (VGG16, ResNet50, Vision Transformer)
- **Métriques de similarité configurables** : Choix entre plusieurs métriques (euclidienne, chi-square, correlation, bhattacharyya, cosinus)
- **Résultats personnalisables** : Sélection du nombre de résultats à afficher (Top-N)
- **Évaluation des performances** : Génération et affichage de courbes de précision-rappel pour évaluer la qualité des résultats
- **Interface web intuitive** : Interface utilisateur simple et efficace basée sur Flask et Bulma CSS

## Modèles et performances

### Modèles supportés

- **VGG16** : Réseau de neurones convolutif profond
- **ResNet50** : Réseau résiduel à 50 couches
- **Vision Transformer (ViT)** : Modèle basé sur l'architecture Transformer

### Métriques de similarité disponibles

- **Euclidienne** : Distance euclidienne entre les vecteurs de caractéristiques
- **Chi-square** : Distance chi-square entre les vecteurs de caractéristiques
- **Corrélation** : Coefficient de corrélation entre les vecteurs
- **Bhattacharyya** : Distance de Bhattacharyya, utile pour comparer des distributions
- **Cosinus** : Similarité cosinus, mesure l'angle entre les vecteurs

### Performances et recommandations

- **Choix du modèle** : 
  - **Vision Transformer (ViT)** : Performance maximale (MAP = 0.9126) avec les métriques cosinus et corrélation
  - **ResNet50** : Excellent compromis performance/efficacité (MAP = 0.8994) avec un temps d'exécution 23% plus rapide (35.01s vs 45.83s pour ViT)
  - **VGG16** : Bonne robustesse (MAP = 0.8765) et stabilité à travers les différentes métriques

- **Choix de la métrique** :
  - **Cosinus** : Performances optimales avec tous les modèles (MAP = 0.9126 avec ViT)
  - **Corrélation** : Résultats similaires au cosinus, particulièrement efficace avec ViT et ResNet50
  - **Bhattacharyya** : Bonnes performances avec VGG16 (0.8749) et ResNet50 (0.8602), mais faible avec ViT (0.1900)
  - **Chi-square** : Efficace avec VGG16 (0.8324), mais déconseillée avec ViT (0.0965)
  - **Euclidienne** : Performance modérée mais constante à travers les modèles (0.77 en moyenne)

- **Recommandations** :
  - Pour une **performance maximale** : ViT avec cosinus ou corrélation
  - Pour le **meilleur équilibre performance/rapidité** : ResNet50 avec cosinus
  - Pour une **robustesse maximale** : VGG16 avec cosinus

## Architecture du projet

```
projet/
├── app/                      # Application Flask
│   ├── static/               # Fichiers statiques
│   │   ├── features/         # Descripteurs pré-calculés
│   │   ├── image.orig/       # Base d'images
│   │   ├── results/          # Résultats de recherche temporaires
│   │   └── uploads/          # Images téléchargées par les utilisateurs
│   ├── templates/            # Templates HTML
│   └── main.py               # Point d'entrée de l'application Flask
├── data/                     # Données brutes et traitées
├── docs/                     # Documentation
├── notebooks/                # Notebooks Jupyter pour l'analyse et le développement
├── results/                  # Résultats d'évaluation et analyses
├── src/                      # Code source du moteur de recherche
│   ├── config.py             # Configuration du système
│   ├── evaluation.py         # Fonctions d'évaluation (précision-rappel)
│   ├── indexing.py           # Fonctions d'indexation des images
│   └── retrieval.py          # Fonctions de recherche et de similarité
├── run_evaluation.py         # Script d'évaluation des performances
├── run_indexing.py           # Script d'indexation des images
├── secure_users.db           # Base de données SQLite pour l'authentification
├── README.md                 # Documentation du projet
├── requirements.txt          # Dépendances Python
├── Dockerfile                # Configuration Docker
└── docker-compose.yml        # Configuration Docker Compose
```

## Prérequis

- Python 3.8+
- PyTorch
- Flask
- Matplotlib
- NumPy
- Pillow
- SQLite (pour l'authentification)

## Installation

### Installation locale

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/romeoibraimovski/image-similarity-search.git
   cd image-similarity-search
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Lancez l'application :
   ```bash
   python app/main.py
   ```

4. Accédez à l'application dans votre navigateur à l'adresse `http://127.0.0.1:8080`

### Installation avec Docker

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/romeoxxiv/image-similarity-search.git
   cd image-similarity-search
   ```

2. Construisez et lancez les conteneurs avec Docker Compose :
   ```bash
   docker-compose up -d
   ```

3. Accédez à l'application dans votre navigateur à l'adresse `http://127.0.0.1:8080`

## Guide utilisateur détaillé

### Création d'un compte et connexion

1. Accédez à la page d'accueil de l'application (`http://votre-adresse:8080`)
2. Si vous n'avez pas encore de compte, cliquez sur "S'inscrire" et remplissez le formulaire d'inscription
   - Choisissez un nom d'utilisateur unique
   - Entrez une adresse email valide
   - Créez un mot de passe sécurisé (minimum 6 caractères)
   - Confirmez votre mot de passe
3. Une fois inscrit, vous serez redirigé vers la page de connexion
4. Entrez vos identifiants pour vous connecter

### Recherche d'images par similarité

1. Après la connexion, vous serez dirigé vers la page de recherche
2. Pour rechercher des images similaires, vous avez deux options :
   - **Télécharger une image** : Cliquez sur "Parcourir" pour sélectionner une image de votre ordinateur
   - **Utiliser une image de la base de données** : Sélectionnez "Image de la base de données" et entrez le numéro de l'image

3. Configurez les paramètres de recherche :
   - **Modèle** : Choisissez le modèle d'extraction de caractéristiques (VGG16, ResNet50, ViT)
   - **Métrique de similarité** : Sélectionnez la métrique à utiliser pour comparer les images
   - **Nombre de résultats** : Définissez combien d'images similaires vous souhaitez voir (Top-N)

4. Cliquez sur "Rechercher" pour lancer la recherche

5. Une fois la recherche terminée, vous verrez :
   - Les informations sur la requête (image requête, modèle, métrique, classe, précision moyenne)
   - La courbe de précision-rappel montrant la performance de la recherche
   - Les images les plus similaires classées par ordre de similarité décroissante
   - Pour chaque image résultat : son score de similarité et sa position dans le classement

6. Enfin, vous pouvez cliquer sur "Nouvelle recherche" pour effectuer une autre requête

## Sécurité

L'application intègre plusieurs mesures de protection :

- **Protection CSRF** : Via `flask_wtf.csrf.CSRFProtect` avec jetons automatiques dans les formulaires
- **Sécurisation des mots de passe** : Hachage avec Werkzeug (`generate_password_hash`/`check_password_hash`)
- **Sessions sécurisées** : Signées cryptographiquement avec clé secrète configurable
- **Contrôle d'accès** : Décorateur `@login_required` sur les routes sensibles
- **Uploads sécurisés** : Validation d'extensions, nettoyage des noms de fichiers et noms uniques
- **Protection XSS** : Échappement automatique des variables dans les templates Jinja2
- **Configuration production** : Mode production avec Gunicorn et Docker

## Déploiement en production

### Déploiement sur une VM

1. Préparez votre VM (Ubuntu 20.04+ recommandé) :
   ```bash
   # Mise à jour du système
   sudo apt update && sudo apt upgrade -y
   
   # Installation de Docker et Docker Compose
   sudo apt install -y docker.io docker-compose
   sudo systemctl enable docker
   sudo systemctl start docker
   sudo usermod -aG docker $USER
   ```

2. Clonez le dépôt sur votre VM :
   ```bash
   git clone https://github.com/romeoibraimovski/image-similarity-search.git
   cd image-similarity-search
   ```

3. Configurez les variables d'environnement :
   ```bash
   # Créez un fichier .env
   echo "SECRET_KEY=votre_cle_secrete_tres_longue_et_aleatoire" > .env
   ```

4. Générez ou transférez les descripteurs d'images (fichiers PKL) :
   ```bash
   # Option 1 : Générer les descripteurs (nécessite les images dans app/static/image.orig/)
   python run_indexing.py
   
   # Option 2 : Si vous avez déjà les descripteurs sur votre machine locale
   # Sur votre machine locale :
   # tar -czvf features.tar.gz app/static/features/*.pkl
   # scp features.tar.gz user@adresse_ip_vm:~/image-similarity-search/
   
   # Sur la VM :
   # tar -xzvf features.tar.gz
   ```

5. Lancez l'application avec Docker Compose :
   ```bash
   docker-compose up -d
   ```

6. Accédez à l'application via l'adresse IP de votre VM sur le port 8080 :
   ```
   http://IP_DE_VOTRE_VM:8080
   ```

### Maintenance et mise à jour

Pour mettre à jour l'application :
```bash
cd image-similarity-search
git pull
docker-compose down
docker-compose up -d --build
```

Pour sauvegarder la base de données :
```bash
# Créez un répertoire de sauvegarde
mkdir -p ~/backups

# Sauvegardez la base de données
cp secure_users.db ~/backups/secure_users_$(date +%Y%m%d).db
```

## Auteurs

- Romeo Ibraimovski
- Maxime Dupuis
