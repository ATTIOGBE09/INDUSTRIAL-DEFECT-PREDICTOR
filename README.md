#  INDUSTRIAL-DEFECT-PREDICTOR

Application interactive pour l’analyse et la prédiction des défauts de production dans un environnement industriel.

---

##  Objectif du projet

L’objectif est d'anticiper l'apparition de défauts dans un processus de production en s'appuyant sur les données collectées (production, qualité, consommation d'énergie, etc.). Grâce à un modèle de Machine Learning, cette solution permet :

- de détecter les conditions propices aux défauts,
- de prédire la probabilité d’un défaut pour une configuration donnée,
- et de générer des recommandations automatiques d’amélioration.

---

##  Méthodologie

Le projet a été réalisé en plusieurs étapes :

1. **Exploration et analyse des données**  
   - Étude des distributions selon le statut de défaut  
   - Identification des variables les plus influentes (MaintenanceHours, QualityScore, etc.)

2. **Préparation des données**  
   - Nettoyage, sélection de variables pertinentes  
   - Transformation des données pour l'entraînement

3. **Modélisation (Random Forest)**  
   - Séparation train/test (80/20)  
   - Validation croisée (5-fold)  
   - Évaluation via matrice de confusion et AUC

4. **Déploiement via Streamlit**  
   - Interface utilisateur simple et ergonomique  
   - Prédiction en temps réel  
   - Génération automatique de recommandations en fonction des inputs

---

##  Fonctionnalités de l'application

-  Exploration visuelle des variables par défaut
-  Prédiction du statut de défaut selon les données saisies
-  Suggestions automatiques d’actions (qualité, maintenance, etc.)
-  Interface claire, interactive et intuitive avec **Streamlit**

---

##  Technologies utilisées

- Python
- Scikit-learn
- Pandas & NumPy
- Streamlit
- Plotly (visualisation)
- Joblib (enregistrement du modèle)

---

## Comment lancer l'application

```bash
# Cloner le dépôt
git clone https://github.com/ATTIOGBE09/INDUSTRIAL-DEFECT-PREDICTOR.git
cd INDUSTRIAL-DEFECT-PREDICTOR

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
