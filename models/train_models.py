"""
train_models.py
---------------
Entraîne les 3 modèles de recommandation, applique une PCA sur les embeddings,
et uploade tous les artefacts sur S3.

Usage :
    python train_models.py --s3-bucket mon-bucket --data-dir ../data
    # Dataset complet (clicks.zip détecté automatiquement) :
    python train_models.py --s3-bucket mon-bucket --data-dir ../data --max-sim-articles 10000
"""

import argparse
import os
import pickle
import zipfile
import numpy as np
import pandas as pd
import boto3
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from implicit.als import AlternatingLeastSquares

# ─────────────────────────────────────────────
# 1. Arguments CLI
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--s3-bucket", required=True, help="Nom du bucket S3")
parser.add_argument(
    "--data-dir",
    default="../news-portal-user-interactions-by-globocom",
    help="Dossier contenant les données source",
)
parser.add_argument(
    "--pca-components", type=int, default=64,
    help="Nombre de composantes PCA pour les embeddings (défaut : 64)",
)
parser.add_argument(
    "--max-sim-articles", type=int, default=10000,
    help="Nb max d'articles pour la matrice de similarité (défaut : 10000). "
         "Limité aux articles les plus cliqués pour tenir en mémoire.",
)
args = parser.parse_args()

DATA_DIR         = args.data_dir
S3_BUCKET        = args.s3_bucket
PCA_N            = args.pca_components
MAX_SIM_ARTICLES = args.max_sim_articles
OUT_DIR          = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. Chargement des données
# ─────────────────────────────────────────────
print("Chargement des données...")
articles_df = pd.read_csv(os.path.join(DATA_DIR, "articles_metadata.csv"))

# Auto-détection : clicks.zip (dataset complet) > clicks_sample.csv
clicks_zip = os.path.join(DATA_DIR, "clicks.zip")
clicks_csv = os.path.join(DATA_DIR, "clicks_sample.csv")

if os.path.exists(clicks_zip):
    print("  Source clics : clicks.zip (dataset complet)")
    dfs = []
    with zipfile.ZipFile(clicks_zip) as z:
        csv_files = sorted(n for n in z.namelist() if n.endswith(".csv"))
        for name in csv_files:
            with z.open(name) as f:
                dfs.append(pd.read_csv(f, usecols=["user_id", "click_article_id"]))
    clicks_df = pd.concat(dfs, ignore_index=True)
else:
    print("  Source clics : clicks_sample.csv")
    clicks_df = pd.read_csv(clicks_csv, usecols=["user_id", "click_article_id"])

path_embedding = os.path.join(DATA_DIR, "articles_embeddings.pickle")
with open(path_embedding, "rb") as f:
    embeddings_raw = pickle.load(f)

# Normalisation : embeddings peut être un ndarray ou un dict
if isinstance(embeddings_raw, np.ndarray):
    article_ids = articles_df["article_id"].tolist()
    embeddings_matrix = embeddings_raw
else:
    article_ids = [int(k) for k in embeddings_raw.keys()]
    embeddings_matrix = np.array(list(embeddings_raw.values()))

print(f"  Articles       : {len(articles_df):,}")
print(f"  Clics          : {len(clicks_df):,}")
print(f"  Embeddings     : {embeddings_matrix.shape}")

# ─────────────────────────────────────────────
# 3. Séparation train / test
# ─────────────────────────────────────────────
train_df, test_df = train_test_split(clicks_df, test_size=0.2, random_state=42)

# ─────────────────────────────────────────────
# 4. Mappings ALS (tous les articles du train)
#    Construction sparse directe (pivot_table dense inutilisable à grande échelle)
# ─────────────────────────────────────────────
user_ids_list    = sorted(train_df["user_id"].unique().tolist())
article_ids_list = sorted(train_df["click_article_id"].unique().tolist())

user_idx    = {uid: i for i, uid in enumerate(user_ids_list)}
article_idx = {aid: i for i, aid in enumerate(article_ids_list)}
idx_article = {i: aid for aid, i in article_idx.items()}

rows = train_df["user_id"].map(user_idx).values
cols = train_df["click_article_id"].map(article_idx).values

# ─────────────────────────────────────────────
# 5. Modèle 1 – ALS (collaborative filtering)
# ─────────────────────────────────────────────
print("\nEntraînement ALS...")
user_item_sparse = csr_matrix(
    (np.ones(len(rows)), (rows, cols)),
    shape=(len(user_ids_list), len(article_ids_list)),
)

als_model = AlternatingLeastSquares(factors=50, iterations=20, regularization=0.1, use_gpu=False)
als_model.fit(user_item_sparse)

als_artifacts = {
    "user_factors": als_model.user_factors,  # shape (n_users, factors)
    "item_factors": als_model.item_factors,  # shape (n_items, factors)
}
print(f"  ALS entraîné  ({len(user_ids_list):,} users × {len(article_ids_list):,} articles)")

# ─────────────────────────────────────────────
# 6. Modèle 2 – Embeddings + PCA (content-based)
# ─────────────────────────────────────────────
print(f"\nPCA sur les embeddings ({embeddings_matrix.shape[1]}D → {PCA_N}D)...")
pca = PCA(n_components=PCA_N, random_state=42)
embeddings_pca = pca.fit_transform(embeddings_matrix).astype(np.float32)
print(f"  Variance expliquée : {pca.explained_variance_ratio_.sum():.2%}")

# ─────────────────────────────────────────────
# 7. Modèle 3 – Similarité item-based
#    Limité aux MAX_SIM_ARTICLES articles les plus cliqués
#    pour éviter une matrice trop grande (N² × 4 octets)
# ─────────────────────────────────────────────
print(f"\nCalcul de la matrice de similarité item-based (top {MAX_SIM_ARTICLES:,} articles)...")

top_sim_aids = (
    train_df["click_article_id"]
    .value_counts()
    .head(MAX_SIM_ARTICLES)
    .index.tolist()
)
sim_article_idx = {aid: i for i, aid in enumerate(top_sim_aids)}
sim_idx_article = {i: aid for aid, i in sim_article_idx.items()}

train_sim = train_df[train_df["click_article_id"].isin(top_sim_aids)]
sim_rows  = train_sim["user_id"].map({uid: i for i, uid in enumerate(sorted(train_sim["user_id"].unique()))}).values
sim_cols  = train_sim["click_article_id"].map(sim_article_idx).values
n_sim_users = train_sim["user_id"].nunique()

sim_user_item = csr_matrix(
    (np.ones(len(sim_rows)), (sim_rows, sim_cols)),
    shape=(n_sim_users, len(top_sim_aids)),
)
item_similarity = cosine_similarity(sim_user_item.T).astype(np.float32)
sim_mb = item_similarity.nbytes / 1e6
print(f"  Matrice similarité : {item_similarity.shape}  ({sim_mb:.0f} Mo)")

# ─────────────────────────────────────────────
# 8. Sauvegarde locale
# ─────────────────────────────────────────────
print("\nSauvegarde des artefacts...")

# Pré-calculer user_clicks : évite de charger pandas dans Lambda
user_clicks = train_df.groupby("user_id")["click_article_id"].apply(list).to_dict()

mappings = {
    "user_ids":                  user_ids_list,
    "article_ids":               article_ids_list,
    "user_idx":                  user_idx,
    "article_idx":               article_idx,
    "idx_article":               idx_article,
    # Embeddings
    "emb_article_ids":           article_ids,
    "article_id_to_emb_index":   {aid: i for i, aid in enumerate(article_ids)},
    # Similarité (sous-ensemble des articles les plus cliqués)
    "sim_article_idx":           sim_article_idx,
    "sim_idx_article":           sim_idx_article,
}

artifacts = {
    "mappings.pkl":        mappings,
    "als_model.pkl":       als_artifacts,
    "embeddings_pca.pkl":  embeddings_pca,
    "item_similarity.pkl": item_similarity,
    "user_clicks.pkl":     user_clicks,
}

for filename, obj in artifacts.items():
    path = os.path.join(OUT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  {filename} → {size_mb:.1f} Mo")

# ─────────────────────────────────────────────
# 9. Évaluation – Recall@10 sur le jeu de test
# ─────────────────────────────────────────────
print("\nÉvaluation Recall@10 sur le jeu de test...")

TOP_N_EVAL = 10
test_ground_truth = test_df.groupby("user_id")["click_article_id"].apply(set).to_dict()

def recall_at_k(reco_ids: set, true_ids: set) -> float:
    return len(reco_ids & true_ids) / len(true_ids) if true_ids else 0.0

# ALS
recalls_als = []
for uid, true_set in test_ground_truth.items():
    if uid not in user_idx:
        continue
    u_vec = als_artifacts["user_factors"][user_idx[uid]]
    scores = als_artifacts["item_factors"] @ u_vec
    seen = [article_idx[a] for a in user_clicks.get(uid, []) if a in article_idx]
    if seen:
        scores = scores.copy()
        scores[seen] = -np.inf
    top = set(idx_article[i] for i in np.argpartition(scores, -TOP_N_EVAL)[-TOP_N_EVAL:])
    recalls_als.append(recall_at_k(top, true_set))

# Embeddings PCA
aid_to_idx_emb = mappings["article_id_to_emb_index"]
emb_ids = mappings["emb_article_ids"]
recalls_emb = []
for uid, true_set in test_ground_truth.items():
    clicked = user_clicks.get(uid, [])
    idxs = [aid_to_idx_emb[a] for a in clicked if a in aid_to_idx_emb]
    if not idxs:
        continue
    profile = embeddings_pca[idxs].mean(axis=0, keepdims=True)
    scores = (embeddings_pca @ profile.T).flatten()
    scores = scores.copy()
    scores[idxs] = -np.inf
    top = set(int(emb_ids[i]) for i in np.argpartition(scores, -TOP_N_EVAL)[-TOP_N_EVAL:])
    recalls_emb.append(recall_at_k(top, true_set))

# Similarité item-based
recalls_sim = []
for uid, true_set in test_ground_truth.items():
    clicked = user_clicks.get(uid, [])
    idxs = [sim_article_idx[a] for a in clicked if a in sim_article_idx]
    if not idxs:
        continue
    scores = item_similarity[idxs].sum(axis=0)
    scores = scores.copy()
    scores[idxs] = -np.inf
    top = set(sim_idx_article[i] for i in np.argpartition(scores, -TOP_N_EVAL)[-TOP_N_EVAL:])
    recalls_sim.append(recall_at_k(top, true_set))

print(f"  ALS             Recall@10 = {np.mean(recalls_als):.4f}  (n={len(recalls_als):,} users)")
print(f"  Embeddings PCA  Recall@10 = {np.mean(recalls_emb):.4f}  (n={len(recalls_emb):,} users)")
print(f"  Similarité      Recall@10 = {np.mean(recalls_sim):.4f}  (n={len(recalls_sim):,} users)")

# ─────────────────────────────────────────────
# 10. Upload sur S3
# ─────────────────────────────────────────────
print(f"\nUpload vers s3://{S3_BUCKET}/models/ ...")
s3 = boto3.client("s3")

for filename in artifacts:
    local_path = os.path.join(OUT_DIR, filename)
    s3_key     = f"models/{filename}"
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"  Uploadé : {s3_key}")

print("\nTerminé.")
