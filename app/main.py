import os
import sys
from flask import Flask, jsonify, request, render_template, redirect, session, flash, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import sqlite3
import time
import uuid
from datetime import datetime
import pickle
from flask_wtf.csrf import CSRFProtect

# Ajouter le répertoire parent au chemin de recherche Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration de Matplotlib pour utiliser un backend non-interactif
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend Agg qui ne nécessite pas de GUI

# --- Local imports ---
from src.retrieval import load_features, search
from src.config import IMAGE_DATASET_PATH, MODELS_TO_INDEX
from src.evaluation import get_image_class, calculate_average_precision

UPLOAD_FOLDER = os.path.join("app", "static", "uploads")

DATABASE = 'secure_users.db'

# Créer le dossier d'upload s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'changez-moi-en-production')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Initialisation de la protection CSRF
csrf = CSRFProtect(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def init_db():
    conn = sqlite3.connect(DATABASE)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def get_user_by_username(username):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    user = conn.execute(
        'SELECT * FROM users WHERE username = ?', (username,)
    ).fetchone()
    conn.close()
    return dict(user) if user else None

def create_user(username, email, password):
    password_hash = generate_password_hash(password)
    conn = sqlite3.connect(DATABASE)
    try:
        conn.execute(
            'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
            (username, email, password_hash)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Vous devez être connecté pour accéder à cette page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

init_db()
print("Chargement des descripteurs en mémoire...")
ALL_FEATURES = load_features()
print("Tous les descripteurs sont chargés.")

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect('/search')
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    
    username = request.form.get('username', '').strip().lower()
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    confirm_password = request.form.get('confirm_password', '')

    if not username or not email or not password:
        flash('Tous les champs sont obligatoires.', 'danger')
        return render_template('register.html')
    
    if len(password) < 6:
        flash('Le mot de passe doit contenir au moins 6 caractères.', 'danger')
        return render_template('register.html')
    
    if password != confirm_password:
        flash('Les mots de passe ne correspondent pas.', 'danger')
        return render_template('register.html')

    if create_user(username, email, password):
        flash('Compte créé avec succès ! Vous pouvez maintenant vous connecter.', 'success')
        return redirect(url_for('login'))
    else:
        flash('Ce nom d\'utilisateur ou email existe déjà.', 'danger')
        return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    username = request.form.get('username', '').strip().lower()
    password = request.form.get('password', '')
    
    if not username or not password:
        flash('Nom d\'utilisateur et mot de passe requis.', 'danger')
        return render_template('login.html')
    
    user = get_user_by_username(username)
    
    if user and check_password_hash(user['password_hash'], password):
        session['user_id'] = username
        session['user_email'] = user['email']
        flash(f'Bienvenue, {username} !', 'success')
        return redirect(url_for('search_page'))
    else:
        flash('Nom d\'utilisateur ou mot de passe incorrect.', 'danger')
        return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('Vous avez été déconnecté.', 'info')
    return redirect(url_for('login'))

@app.route('/search', methods=['GET'])
@login_required
def search_page():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
@login_required
def search_images():
    try:
        # Récupération de la source de l'image et de la classe
        image_source = request.form.get('image_source', 'external')
        image_class = int(request.form.get('image_class', 0))
        
        # Traitement différent selon la source de l'image
        if image_source == 'external':
            # Récupération du fichier image externe
            if 'image' not in request.files:
                flash('Aucun fichier sélectionné', 'danger')
                return redirect(url_for('search_page'))
            
            file = request.files['image']
            if file.filename == '':
                flash('Aucun fichier sélectionné', 'danger')
                return redirect(url_for('search_page'))
            
            # Vérification de l'extension du fichier
            if not allowed_file(file.filename):
                flash('Format de fichier non supporté. Utilisez JPG, JPEG ou PNG.', 'danger')
                return redirect(url_for('search_page'))
            
            # Sauvegarde temporaire de l'image
            filename = secure_filename(file.filename)
            # Créer un nom de fichier unique avec timestamp pour éviter les collisions
            unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}_{filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(save_path)
            
        else:  # image_source == 'database'
            # Récupération de l'image de la base de données
            database_image_id = request.form.get('database_image')
            if not database_image_id:
                flash('Veuillez sélectionner une image de la base de données', 'danger')
                return redirect(url_for('search_page'))
            
            # Construire le chemin vers l'image de la base de données
            image_number = int(database_image_id)
            save_path = os.path.join(IMAGE_DATASET_PATH, f"{image_number}.jpg")
            
            # Vérifier si l'image existe
            if not os.path.exists(save_path):
                flash(f'Image {image_number}.jpg non trouvée dans la base de données', 'danger')
                return redirect(url_for('search_page'))
        
        # Récupération des paramètres de recherche
        model = request.form.get('model', 'resnet50')
        similarity = request.form.get('similarity', 'cosine')
        top_n = int(request.form.get('top_n', 5))
        
        # Recherche des images similaires
        results = search(save_path, model, ALL_FEATURES, distance_metric=similarity, top_n=top_n)
        
        # Stocker tous les résultats pour la courbe R/P (jusqu'à 1000)
        all_results = search(save_path, model, ALL_FEATURES, distance_metric=similarity, top_n=1000)
        
        # Convertir les résultats complets en format sérialisable et corriger les chemins d'images
        serializable_results = []
        for path, score in results:
            # Convertir les valeurs numpy en types Python natifs
            if hasattr(score, 'item'):
                score = score.item()
            
            # Corriger le chemin de l'image pour qu'il soit servi par Flask
            # Si le chemin contient 'image.orig', le convertir en URL statique
            if 'image.orig' in path:
                # Extraire le nom du fichier
                filename = os.path.basename(path)
                # Créer un chemin relatif au dossier statique
                corrected_path = url_for('static', filename=f'image.orig/{filename}')
            else:
                corrected_path = path
                
            serializable_results.append((corrected_path, float(score)))
        
        serializable_all_results = []
        for path, score in all_results:
            # Convertir les valeurs numpy en types Python natifs
            if hasattr(score, 'item'):
                score = score.item()
            
            # Pour all_results, on garde le chemin original pour l'évaluation
            serializable_all_results.append((path, float(score)))
        
        # Au lieu de stocker tous les résultats dans la session, stockons-les dans un fichier temporaire
        # et mettons juste le chemin du fichier dans la session
        results_dir = os.path.join(app.static_folder, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Créer un identifiant unique pour cette recherche
        search_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}"
        
        # Stocker les informations minimales dans la session
        session['query_path'] = save_path
        session['model'] = model
        session['similarity'] = similarity
        session['top_n'] = top_n
        session['search_id'] = search_id
        session['image_class'] = image_class  # Stocker la classe de l'image pour la courbe R/P
        session['image_source'] = image_source  # Stocker la source de l'image
        
        # Stocker les résultats dans des fichiers temporaires
        with open(os.path.join(results_dir, f"{search_id}_results.pkl"), 'wb') as f:
            pickle.dump(serializable_results, f)
        
        with open(os.path.join(results_dir, f"{search_id}_all_results.pkl"), 'wb') as f:
            pickle.dump(serializable_all_results, f)
        
        return redirect(url_for('results_page'))
        
    except Exception as e:
        flash(f'Erreur lors de la recherche: {str(e)}', 'danger')
        return render_template('index.html')

@app.route('/results')
@login_required
def results_page():
    try:
        # Récupérer les informations de la session
        query_path = session.get('query_path')
        model = session.get('model')
        similarity = session.get('similarity')
        top_n = session.get('top_n')
        search_id = session.get('search_id')
        image_class = session.get('image_class')
        image_source = session.get('image_source')
        
        if not all([query_path, model, similarity, top_n, search_id]):
            flash('Session expirée ou informations manquantes', 'danger')
            return redirect(url_for('search_page'))
        
        # Traduction des métriques de similarité en français
        similarity_translations = {
            'euclidean': 'Distance Euclidienne',
            'chi_square': 'Chi-carré',
            'correlation': 'Corrélation',
            'bhattacharyya': 'Bhattacharyya',
            'cosine': 'Cosinus'
        }
        similarity_fr = similarity_translations.get(similarity, similarity)
        
        # Récupérer les résultats depuis les fichiers temporaires
        results_dir = os.path.join(app.static_folder, 'results')
        
        with open(os.path.join(results_dir, f"{search_id}_results.pkl"), 'rb') as f:
            results = pickle.load(f)
        
        with open(os.path.join(results_dir, f"{search_id}_all_results.pkl"), 'rb') as f:
            all_results = pickle.load(f)
        
        # Générer la courbe rappel-précision
        pr_curve_path = None
        average_precision = None
        
        if image_class is not None:
            # Calculer le nombre total d'images pertinentes pour cette classe
            total_relevant_docs = 0
            for filename in os.listdir(IMAGE_DATASET_PATH):
                if filename.endswith('.jpg') and int(filename.split('.')[0]) // 100 == image_class:
                    total_relevant_docs += 1
            
            # Calculer la courbe rappel-précision
            ap, (recall_points, precision_points) = calculate_average_precision(all_results, image_class, total_relevant_docs)
            average_precision = round(ap * 100, 2)  # Convertir en pourcentage et arrondir
            
            # Générer la courbe
            import matplotlib.pyplot as plt
            import time
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall_points, precision_points, 'b-', linewidth=2)
            plt.xlabel('Rappel')
            plt.ylabel('Précision')
            plt.title(f'Courbe Rappel/Précision - AP: {average_precision}%')
            plt.grid(True)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            
            # Sauvegarder la courbe
            pr_curve_filename = f"pr_curve_{search_id}.png"
            pr_curve_path = os.path.join('results', pr_curve_filename)
            plt.savefig(os.path.join(app.static_folder, pr_curve_path))
            plt.close()
        
        # Déterminer le nom de la classe à partir du numéro
        class_names = {
            0: "Personnes",
            1: "Plages",
            2: "Architecture",
            3: "Bus",
            4: "Dinosaures",
            5: "Éléphants",
            6: "Fleurs",
            7: "Chevaux",
            8: "Montagnes",
            9: "Nourriture"
        }
        class_name = class_names.get(image_class)
        
        # Préparer le chemin de l'image requête pour l'affichage
        display_query_path = query_path
        if image_source == 'database':
            # Si c'est une image de la base de données, on construit l'URL correcte
            image_number = os.path.basename(query_path).split('.')[0]
            display_query_path = url_for('static', filename=f'image.orig/{image_number}.jpg')
        elif query_path and os.path.exists(query_path) and 'uploads' in query_path:
            # Si c'est une image téléchargée dans le dossier uploads
            filename = os.path.basename(query_path)
            display_query_path = url_for('static', filename=f'uploads/{filename}')
        elif '/static/' in query_path:
            # Si l'image est dans le dossier static, on extrait le chemin relatif
            static_dir = os.path.join(app.root_path, 'static')
            if query_path.startswith(static_dir):
                rel_path = os.path.relpath(query_path, static_dir)
                display_query_path = url_for('static', filename=rel_path)
            else:
                display_query_path = url_for('static', filename=os.path.basename(query_path))
        
        return render_template(
            'results.html',
            results=results,
            query_path=display_query_path,
            model=model,
            similarity=similarity_fr,
            top_n=top_n,
            pr_curve_path=pr_curve_path,
            average_precision=average_precision,
            image_class=image_class,
            class_name=class_name
        )
        
    except Exception as e:
        flash(f'Erreur lors de l\'affichage des résultats: {str(e)}', 'danger')
        return redirect(url_for('search_page'))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
