"""
train_models.py
---------------
Entraîne les 3 modèles de recommandation, applique une PCA sur les embeddings,
et uploade tous les artefacts sur S3.

Usage :
    python train_models.py --s3-bucket mon-bucket --data-dir ../data
"""

import argparse
import os
import pickle
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
args = parser.parse_args()

DATA_DIR   = args.data_dir
S3_BUCKET  = args.s3_bucket
PCA_N      = args.pca_components
OUT_DIR    = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. Chargement des données
# ─────────────────────────────────────────────
print("Chargement des données...")
articles_df = pd.read_csv(os.path.join(DATA_DIR, "articles_metadata.csv"))
clicks_df   = pd.read_csv(os.path.join(DATA_DIR, "clicks_sample.csv"))

path_embedding = os.path.join(DATA_DIR, "articles_embeddings.pickle")
with open(path_embedding, "rb") as f:
    embeddings_raw = pickle.load(f)

# Normalisation : embeddings peut être un ndarray ou un dict
if isinstance(embeddings_raw, np.ndarray):
    article_ids = articles_df["article_id"].tolist()
    embeddings_matrix = embeddings_raw
else:
    article_ids = list(embeddings_raw.keys())
    embeddings_matrix = np.array(list(embeddings_raw.values()))

print(f"  Articles : {len(articles_df)}")
print(f"  Clics    : {len(clicks_df)}")
print(f"  Embeddings shape : {embeddings_matrix.shape}")

# ─────────────────────────────────────────────
# 3. Séparation train / test
# ─────────────────────────────────────────────
train_df, test_df = train_test_split(clicks_df, test_size=0.2, random_state=42)

# ─────────────────────────────────────────────
# 4. Mappings communs
# ─────────────────────────────────────────────
user_item_matrix = train_df.pivot_table(
    index="user_id", columns="click_article_id", aggfunc="size", fill_value=0
)
user_ids_list    = user_item_matrix.index.tolist()
article_ids_list = user_item_matrix.columns.tolist()

user_idx   = {uid: i for i, uid in enumerate(user_ids_list)}
article_idx = {aid: i for i, aid in enumerate(article_ids_list)}
idx_article = {i: aid for aid, i in article_idx.items()}

mappings = {
    "user_ids":      user_ids_list,
    "article_ids":   article_ids_list,
    "user_idx":      user_idx,
    "article_idx":   article_idx,
    "idx_article":   idx_article,
    # Pour les embeddings, mapping article_id → index dans la matrice
    "emb_article_ids":   article_ids,
    "article_id_to_emb_index": {aid: i for i, aid in enumerate(article_ids)},
}

# ─────────────────────────────────────────────
# 5. Modèle 1 – ALS (collaborative filtering)
# ─────────────────────────────────────────────
print("\nEntraînement ALS...")
user_item_sparse = csr_matrix(user_item_matrix.values)

als_model = AlternatingLeastSquares(factors=50, iterations=20, regularization=0.1)
als_model.fit(user_item_sparse)

print("  ALS entraîné.")

# ─────────────────────────────────────────────
# 6. Modèle 2 – Embeddings + PCA (content-based)
# ─────────────────────────────────────────────
print(f"\nPCA sur les embeddings ({embeddings_matrix.shape[1]}D → {PCA_N}D)...")
pca = PCA(n_components=PCA_N, random_state=42)
embeddings_pca = pca.fit_transform(embeddings_matrix).astype(np.float32)
print(f"  Variance expliquée : {pca.explained_variance_ratio_.sum():.2%}")

# ─────────────────────────────────────────────
# 7. Modèle 3 – Similarité item-based
# ─────────────────────────────────────────────
print("\nCalcul de la matrice de similarité item-based...")
item_user_sparse = user_item_sparse.T
item_similarity  = cosine_similarity(item_user_sparse).astype(np.float32)
print(f"  Matrice similarité : {item_similarity.shape}")

# ─────────────────────────────────────────────
# 8. Sauvegarde locale
# ─────────────────────────────────────────────
print("\nSauvegarde des artefacts...")

artifacts = {
    "mappings.pkl":         mappings,
    "als_model.pkl":        als_model,
    "embeddings_pca.pkl":   {"embeddings": embeddings_pca, "pca": pca},
    "item_similarity.pkl":  item_similarity,
    "train_df.pkl":         train_df,
}

for filename, obj in artifacts.items():
    path = os.path.join(OUT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  {filename} → {size_mb:.1f} Mo")

# ─────────────────────────────────────────────
# 9. Upload sur S3
# ─────────────────────────────────────────────
print(f"\nUpload vers s3://{S3_BUCKET}/models/ ...")
s3 = boto3.client("s3")

for filename in artifacts:
    local_path = os.path.join(OUT_DIR, filename)
    s3_key     = f"models/{filename}"
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"  Uploadé : {s3_key}")

print("\nTerminé.")
