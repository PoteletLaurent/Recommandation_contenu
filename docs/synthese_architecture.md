# Synthèse architecture – Système de recommandation d'articles

---

## 1. Description fonctionnelle

### Contexte
L'application permet de tester un système de recommandation d'articles à des particuliers. En l'absence de données utilisateurs propriétaires, le développement du MVP s'appuie sur le dataset public **Globo.com** (interactions utilisateurs sur un portail d'actualités brésilien, 2017).

### Fonctionnalité MVP
> "En tant qu'utilisateur, je reçois une sélection de 5 articles recommandés."

L'interface permet de :
- Sélectionner un utilisateur parmi les utilisateurs connus
- Choisir le modèle de recommandation
- Visualiser les 5 articles recommandés avec leur catégorie, leur longueur et un indicateur de cohérence avec l'historique de l'utilisateur

---

## 2. Architecture technique actuelle (MVP)

### Schéma

```
┌─────────────────────────────────────────────────────────────────┐
│  Utilisateur                                                    │
│       │                                                         │
│       ▼                                                         │
│  Streamlit (EC2 – 13.50.152.40:8501)                           │
│  - Liste des utilisateurs                                       │
│  - Choix du modèle                                             │
│  - Affichage des 5 recommandations                             │
│       │                                                         │
│       │  boto3 (appel direct)                                  │
│       ▼                                                         │
│  AWS Lambda  (p10-recommandation-handler, eu-west-3)           │
│  - Reçoit : user_id + model                                    │
│  - Charge les artefacts depuis S3 (cache mémoire)             │
│  - Calcule et retourne les 5 articles                          │
│       │                                                         │
│       │  Download au 1er appel                                 │
│       ▼                                                         │
│  AWS S3  (p10-recommandation-models)                           │
│  ├── mappings.pkl          (index user/article)                │
│  ├── als_model.pkl         (facteurs ALS numpy)               │
│  ├── embeddings_pca.pkl    (vecteurs articles 64D)            │
│  ├── item_similarity.pkl   (matrice similarité)               │
│  └── user_clicks.pkl       (historique utilisateurs)          │
└─────────────────────────────────────────────────────────────────┘
```

### Composants

| Composant | Technologie | Rôle |
|---|---|---|
| Interface | Streamlit sur EC2 | Sélection utilisateur/modèle, affichage résultats |
| Moteur de recommandation | AWS Lambda (Python 3.11) | Calcul des recommandations |
| Stockage artefacts | AWS S3 | Modèles entraînés et données pré-calculées |
| Entraînement | Script Python local | Génère et uploade les artefacts |

### Système de recommandation – 3 modèles

#### Modèle 1 – ALS (Alternating Least Squares)
- **Type** : Filtrage collaboratif
- **Principe** : Factorise la matrice utilisateurs × articles en deux matrices de facteurs latents. Recommande les articles dont les facteurs sont les plus proches de ceux de l'utilisateur.
- **Rappel@10** : 0.11 | **F1** : 0.03
- **Limite** : Ne fonctionne pas pour les nouveaux utilisateurs (cold start)

#### Modèle 2 – Embeddings PCA (Content-based)
- **Type** : Basé sur le contenu
- **Principe** : Chaque article est représenté par un vecteur de 250 dimensions (généré depuis le contenu). Une PCA réduit ces vecteurs à 64 dimensions (97.21% de variance conservée). Le profil d'un utilisateur est la moyenne des vecteurs de ses articles lus. On recommande les articles dont le vecteur est le plus proche du profil.
- **Rappel@10** : 0.21 | **F1** : 0.06
- **Avantage** : Fonctionne pour les nouveaux articles (cold start article)

#### Modèle 3 – Similarité item-based
- **Type** : Filtrage collaboratif item-based
- **Principe** : Calcule la similarité cosinus entre articles à partir des interactions utilisateurs. Recommande les articles les plus souvent consultés conjointement à ceux déjà lus.
- **Rappel@10** : 0.34 | **F1** : 0.07
- **Meilleur modèle sur ce dataset**

---

## 3. Architecture cible – Prise en compte des nouveaux utilisateurs et articles

### Problématique : le cold start

Le cold start désigne la situation où le système ne dispose d'aucun historique pour faire une recommandation :
- **Nouvel utilisateur** : aucun clic enregistré
- **Nouvel article** : jamais consulté par personne

Les modèles ALS et similarité item-based échouent dans ces deux cas car ils reposent entièrement sur les interactions passées.

### Architecture cible proposée

```
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│   Nouvel article publié                                               │
│         │                                                             │
│         ▼                                                             │
│   Génération embedding (modèle NLP)  ──►  S3 (mise à jour auto)     │
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────┐    │
│   │  Streamlit                                                   │    │
│   │       │                                                      │    │
│   │       ▼                                                      │    │
│   │  Lambda  ──►  Routeur de stratégie                          │    │
│   │                    │                                         │    │
│   │         ┌──────────┼──────────────┐                         │    │
│   │         ▼          ▼              ▼                         │    │
│   │   Utilisateur   Utilisateur   Nouvel        Nouvel          │    │
│   │   connu avec    connu sans    utilisateur   article         │    │
│   │   historique    historique                                  │    │
│   │       │              │            │              │          │    │
│   │       ▼              ▼            ▼              ▼          │    │
│   │  ALS ou         Popularité   Onboarding    Embeddings       │    │
│   │  Similarité     + contenu    (préférences) (contenu)        │    │
│   └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│   Réentraînement périodique (AWS EventBridge + Lambda)               │
│         │                                                             │
│         ▼                                                             │
│   Mise à jour des artefacts S3  ──►  Lambda recharge au prochain     │
│                                       appel (cache invalidé)         │
└───────────────────────────────────────────────────────────────────────┘
```

### Stratégies par cas

| Cas | Stratégie | Implémentation |
|---|---|---|
| Utilisateur connu | ALS ou similarité item-based | Déjà en place |
| Utilisateur sans historique | Popularité globale + préférences de catégories déclarées | Onboarding à l'inscription |
| Nouvel article | Embeddings (contenu uniquement) | Génération embedding à la publication |
| Utilisateur avec peu d'historique | Hybride : embeddings + popularité | Seuil : < 3 articles lus |

### Réentraînement continu

Pour intégrer les nouveaux utilisateurs et articles au fil du temps :

1. **AWS EventBridge** déclenche un réentraînement périodique (ex : toutes les 24h)
2. Une **Lambda de réentraînement** (ou un job ECS) relance `train_models.py`
3. Les nouveaux artefacts sont uploadés sur **S3**
4. La Lambda de recommandation recharge automatiquement les artefacts au prochain appel

### Évolution des embeddings

Pour générer automatiquement les embeddings des nouveaux articles à la publication :
- Un modèle NLP léger (ex : sentence-transformers) génère le vecteur de l'article
- Ce vecteur est ajouté à la matrice d'embeddings sur S3
- Aucun réentraînement complet n'est nécessaire pour ce cas

---

## 4. Limites du MVP actuel

| Limite | Impact | Solution cible |
|---|---|---|
| Dataset anonymisé (pas de titres) | Affichage peu lisible | Utiliser des données propriétaires |
| Pas de réentraînement automatique | Modèle statique | EventBridge + Lambda |
| Cold start non géré | Nouveaux users sans recommandation | Stratégie popularité + onboarding |
| Streamlit non persistant (redémarrage EC2) | Perte du service | Supervisor ou systemd |
