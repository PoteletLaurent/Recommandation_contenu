"""
common.py
---------
Utilitaires partagés par les scripts d'entraînement (train_*.py),
de préparation (prepare_data.py) et d'évaluation (evaluate.py).

Convention :
- Chaque script peut être exécuté indépendamment.
- Le split train/test est déterministe (random_state=42) → tous les scripts
  obtiennent exactement le même test set, garantissant la cohérence des
  Recall@10 et des mappings construits depuis train_df.
"""

import os
import pickle
import zipfile

import boto3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OUT_DIR      = "artifacts"
S3_PREFIX    = "models/"
RANDOM_STATE = 42
TEST_SIZE    = 0.2


# ──────────────────────────────────────────────
# Chargement des données source
# ──────────────────────────────────────────────

def load_dataset(data_dir: str) -> dict:
    """Charge articles, clics (clicks.zip si présent, sinon clicks_sample.csv)
    et embeddings articles."""
    print("Chargement des données...")
    articles_df = pd.read_csv(os.path.join(data_dir, "articles_metadata.csv"))

    clicks_zip = os.path.join(data_dir, "clicks.zip")
    clicks_csv = os.path.join(data_dir, "clicks_sample.csv")

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

    path_embedding = os.path.join(data_dir, "articles_embeddings.pickle")
    with open(path_embedding, "rb") as f:
        embeddings_raw = pickle.load(f)

    if isinstance(embeddings_raw, np.ndarray):
        article_ids_emb   = articles_df["article_id"].tolist()
        embeddings_matrix = embeddings_raw
    else:
        article_ids_emb   = [int(k) for k in embeddings_raw.keys()]
        embeddings_matrix = np.array(list(embeddings_raw.values()))

    print(f"  Articles       : {len(articles_df):,}")
    print(f"  Clics          : {len(clicks_df):,}")
    print(f"  Embeddings     : {embeddings_matrix.shape}")

    return {
        "articles_df":       articles_df,
        "clicks_df":         clicks_df,
        "article_ids_emb":   article_ids_emb,
        "embeddings_matrix": embeddings_matrix,
    }


def split_clicks(clicks_df: pd.DataFrame):
    """Split déterministe (random_state=42) → garantit le même test set
    entre tous les scripts."""
    return train_test_split(clicks_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# ──────────────────────────────────────────────
# Construction des mappings
# ──────────────────────────────────────────────

def build_als_mappings(train_df: pd.DataFrame) -> dict:
    """Mappings utilisés par ALS (couvre tous les articles présents dans train)."""
    user_ids_list    = sorted(train_df["user_id"].unique().tolist())
    article_ids_list = sorted(train_df["click_article_id"].unique().tolist())
    user_idx    = {uid: i for i, uid in enumerate(user_ids_list)}
    article_idx = {aid: i for i, aid in enumerate(article_ids_list)}
    idx_article = {i: aid for aid, i in article_idx.items()}
    return {
        "user_ids":    user_ids_list,
        "article_ids": article_ids_list,
        "user_idx":    user_idx,
        "article_idx": article_idx,
        "idx_article": idx_article,
    }


def build_similarity_mappings(train_df: pd.DataFrame, max_articles: int):
    """Mappings dédiés à la similarité item-based : top N articles les plus cliqués."""
    top_aids = (
        train_df["click_article_id"]
        .value_counts()
        .head(max_articles)
        .index.tolist()
    )
    sim_article_idx = {aid: i for i, aid in enumerate(top_aids)}
    sim_idx_article = {i: aid for aid, i in sim_article_idx.items()}
    return sim_article_idx, sim_idx_article, top_aids


def build_embeddings_mappings(article_ids_emb: list) -> dict:
    return {
        "emb_article_ids":         article_ids_emb,
        "article_id_to_emb_index": {aid: i for i, aid in enumerate(article_ids_emb)},
    }


def build_user_clicks(train_df: pd.DataFrame) -> dict:
    """Pré-calcul user_id → [article_ids] : évite la dépendance pandas dans Lambda."""
    return train_df.groupby("user_id")["click_article_id"].apply(list).to_dict()


# ──────────────────────────────────────────────
# IO artefacts
# ──────────────────────────────────────────────

def save_artifact(obj, filename: str, out_dir: str = OUT_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  {filename} → {size_mb:.1f} Mo")
    return path


def upload_to_s3(bucket: str, filename: str, out_dir: str = OUT_DIR,
                 prefix: str = S3_PREFIX) -> None:
    s3 = boto3.client("s3")
    local_path = os.path.join(out_dir, filename)
    s3_key     = f"{prefix}{filename}"
    s3.upload_file(local_path, bucket, s3_key)
    print(f"  Uploadé : s3://{bucket}/{s3_key}")


def load_artifact(filename: str, bucket: str = None, out_dir: str = OUT_DIR,
                  prefix: str = S3_PREFIX):
    """Charge un artefact local. Si absent et bucket fourni, le télécharge depuis S3."""
    path = os.path.join(out_dir, filename)
    if not os.path.exists(path):
        if not bucket:
            raise FileNotFoundError(
                f"{path} introuvable et aucun --s3-bucket fourni pour le télécharger."
            )
        os.makedirs(out_dir, exist_ok=True)
        s3 = boto3.client("s3")
        print(f"  Téléchargement s3://{bucket}/{prefix}{filename}{os.linesep}             → {path}")
        s3.download_file(bucket, f"{prefix}{filename}", path)
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────
# Métrique
# ──────────────────────────────────────────────

def recall_at_k(reco_ids: set, true_ids: set) -> float:
    return len(reco_ids & true_ids) / len(true_ids) if true_ids else 0.0
