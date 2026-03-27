"""
handler.py  –  AWS Lambda
--------------------------
Reçoit un user_id et un nom de modèle, retourne les 5 articles recommandés.

Payload attendu (JSON) :
    { "user_id": 123456, "model": "als" }   # model : "als" | "embeddings" | "similarity"

Réponse :
    { "user_id": 123456, "model": "als", "recommendations": [id1, id2, id3, id4, id5] }

Variables d'environnement Lambda à configurer :
    S3_BUCKET  – nom du bucket S3 contenant les artefacts
    S3_PREFIX  – préfixe (défaut : "models/")
"""

import json
import os
import pickle
import boto3
import numpy as np

# ─────────────────────────────────────────────
# Chargement des artefacts (une seule fois au démarrage du container)
# ─────────────────────────────────────────────
S3_BUCKET = os.environ["S3_BUCKET"]
S3_PREFIX = os.environ.get("S3_PREFIX", "models/")

s3 = boto3.client("s3")
_cache = {}


def _load(filename: str):
    """Charge un fichier pickle depuis S3 (mis en cache en mémoire)."""
    if filename not in _cache:
        tmp_path = f"/tmp/{filename}"
        s3.download_file(S3_BUCKET, S3_PREFIX + filename, tmp_path)
        with open(tmp_path, "rb") as f:
            _cache[filename] = pickle.load(f)
    return _cache[filename]


def _get_mappings():
    return _load("mappings.pkl")


# ─────────────────────────────────────────────
# Fonctions de recommandation
# ─────────────────────────────────────────────

def recommend_als(user_id: int, top_n: int = 5) -> list:
    als_model = _load("als_model.pkl")
    mappings  = _get_mappings()

    user_idx  = mappings["user_idx"]
    idx_article = mappings["idx_article"]

    if user_id not in user_idx:
        return _fallback(top_n)

    u_idx = user_idx[user_id]
    ids, _ = als_model.recommend(
        u_idx,
        als_model.user_factors[u_idx],   # vecteur utilisateur
        N=top_n,
        filter_already_liked_items=True,
    )
    return [idx_article[i] for i in ids]


def recommend_embeddings(user_id: int, top_n: int = 5) -> list:
    emb_data  = _load("embeddings_pca.pkl")
    mappings  = _get_mappings()
    train_df  = _load("train_df.pkl")

    embeddings = emb_data["embeddings"]
    aid_to_idx = mappings["article_id_to_emb_index"]
    emb_article_ids = mappings["emb_article_ids"]

    clicked = train_df[train_df["user_id"] == user_id]["click_article_id"].tolist()
    indices = [aid_to_idx[aid] for aid in clicked if aid in aid_to_idx]

    if not indices:
        return _fallback(top_n)

    user_profile = embeddings[indices].mean(axis=0, keepdims=True)
    scores = (embeddings @ user_profile.T).flatten()

    # Exclure déjà vus
    seen_indices = set(indices)
    scores[list(seen_indices)] = -np.inf

    top_indices = np.argpartition(scores, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [emb_article_ids[i] for i in top_indices]


def recommend_similarity(user_id: int, top_n: int = 5) -> list:
    item_sim  = _load("item_similarity.pkl")
    mappings  = _get_mappings()
    train_df  = _load("train_df.pkl")

    article_idx  = mappings["article_idx"]
    idx_article  = mappings["idx_article"]

    clicked = train_df[train_df["user_id"] == user_id]["click_article_id"].tolist()
    indices = [article_idx[aid] for aid in clicked if aid in article_idx]

    if not indices:
        return _fallback(top_n)

    scores = item_sim[indices].sum(axis=0)
    scores[indices] = -np.inf  # exclure déjà vus

    top_indices = np.argpartition(scores, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [idx_article[i] for i in top_indices]


def _fallback(top_n: int) -> list:
    """Retourne les articles les plus populaires si l'utilisateur est inconnu."""
    mappings = _get_mappings()
    return mappings["article_ids"][:top_n]


# ─────────────────────────────────────────────
# Handler Lambda
# ─────────────────────────────────────────────

MODELS = {
    "als":        recommend_als,
    "embeddings": recommend_embeddings,
    "similarity": recommend_similarity,
}


def lambda_handler(event, context):
    try:
        user_id    = int(event["user_id"])
        model_name = event.get("model", "similarity").lower()

        if model_name not in MODELS:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Modèle inconnu : {model_name}. Valeurs acceptées : {list(MODELS)}"}),
            }

        recommendations = MODELS[model_name](user_id)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "user_id":        user_id,
                "model":          model_name,
                "recommendations": recommendations,
            }),
        }

    except KeyError as e:
        return {"statusCode": 400, "body": json.dumps({"error": f"Paramètre manquant : {e}"})}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
