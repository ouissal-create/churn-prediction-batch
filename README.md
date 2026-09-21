# Predictive Churn Analytics - Batch Processing

Plateforme décisionnelle d'analyse prédictive du risque d'attrition client (Churn) pour le secteur bancaire. Ce projet démontre la capacité à transformer un modèle de Machine Learning en un outil opérationnel de batch processing.

##  Fonctionnalités Clés

- **Batch Processing** : Analyse de bases clients complètes (CSV/Excel) en une seule opération, contrairement aux approches unitaires.
- **Scoring Automatique** : Prédiction du risque via un modèle Random Forest optimisé (Accuracy: 86.9%).
- **Segmentation Intelligente** : Classification automatique en 3 niveaux (Élevé, Modéré, Faible) avec seuils configurables.
- **Exports Opérationnels** : Génération de 4 rapports Excel stylisés (Analyse complète, Top 50, Segments, Stats) prêts pour les équipes commerciales.
- **Interface Moderne** : Design système "Carte" inspiré des plateformes SaaS, avec KPIs interactifs et graphiques Plotly.

##  Stack Technique

| Catégorie | Technologies |
| :--- | :--- |
| **Langage** | Python 3.11 |
| **Web App** | Streamlit (Batch Processing) |
| **Machine Learning** | Scikit-learn (Random Forest), Pandas, NumPy |
| **Visualisation** | Plotly Express (Graphiques interactifs) |
| **Export Données** | OpenPyXL (Génération Excel stylisée) |
| **Déploiement** | GitHub + Streamlit Community Cloud |

##  Architecture du Projet

```text
churn-prediction-batch/
── app.py                  # Application Streamlit principale (Batch Processing)
├── requirements.txt        # Dépendances Python
├── rf_model.pkl            # Modèle Random Forest entraîné (19 Mo)
── scaler.pkl              # StandardScaler pour le preprocessing
├── features_order.pkl      # Ordre des features pour garantir la cohérence
├── Churn_Modelling.csv     # Dataset d'entraînement original (Kaggle)
└── clients.csv             # Exemple de fichier d'import batch
```

##  Installation & Lancement Local

```bash
# 1. Cloner le dépôt
git clone https://github.com/ouissal-create/churn-prediction-batch.git
cd churn-prediction-batch

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
venv\Scripts\activate  # Sur Windows
# source venv/bin/activate  # Sur Mac/Linux

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
