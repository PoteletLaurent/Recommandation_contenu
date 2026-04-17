"""
handler.py  –  AWS Lambda
--------------------------
Reçoit un user_id et un nom de modèle, retourne les 5 articles recommandés.

Payload attendu (JSON) :
    { "user_id": 123456, "model": "als" }   # model : "als" | "embeddings" | "similarity"

Réponse :
    { "user_id": 123456, "model": "als", "recommendations": [id1, id2, id3, id4, id5] }

Variables d'environnement Lambda :
    S3_BUCKET  – nom du bucket S3 contenant les artefacts
    S3_PREFIX  – préfixe (défaut : "models/")

Dépendances : numpy uniquement (boto3 est fourni par le runtime Lambda)
"""

import json
import os
import pickle
import boto3
import numpy as np

S3_BUCKET = os.environ["S3_BUCKET"]
S3_PREFIX = os.environ.get("S3_PREFIX", "models/")

s3     = boto3.client("s3")
_cache = {}


def _load(filename: str):
    if filename not in _cache:
        tmp_path = f"/tmp/{filename}"
        s3.download_file(S3_BUCKET, S3_PREFIX + filename, tmp_path)
        with open(tmp_path, "rb") as f:
            _cache[filename] = pickle.load(f)
    return _cache[filename]


# ─────────────────────────────────────────────
# Helpers communs
# ─────────────────────────────────────────────

def _seen_indices(user_id, article_idx):
    """Indices des articles déjà vus par l'utilisateur."""
    user_clicks = _load("user_clicks.pkl")
    seen_aids   = user_clicks.get(user_id, [])
    return [article_idx[a] for a in seen_aids if a in article_idx]


def _fallback(top_n: int) -> list:
    return _load("mappings.pkl")["article_ids"][:top_n]


def _top_n(scores, idx_article, seen_indices, top_n):
    scores = scores.copy()
    if seen_indices:
        scores[seen_indices] = -np.inf
    top = np.argpartition(scores, -top_n)[-top_n:]
    top = top[np.argsort(scores[top])[::-1]]
    return [idx_article[i] for i in top]


# ─────────────────────────────────────────────
# Modèles
# ─────────────────────────────────────────────

def recommend_als(user_id: int, top_n: int = 5) -> list:
    als         = _load("als_model.pkl")
    mappings    = _load("mappings.pkl")
    user_idx    = mappings["user_idx"]
    idx_article = mappings["idx_article"]

    if user_id not in user_idx:
        return _fallback(top_n)

    u_vec  = als["user_factors"][user_idx[user_id]]
    scores = als["item_factors"] @ u_vec
    return _top_n(scores, idx_article, _seen_indices(user_id, mappings["article_idx"]), top_n)


def recommend_embeddings(user_id: int, top_n: int = 5) -> list:
    embeddings  = _load("embeddings_pca.pkl")   # numpy array directement
    mappings    = _load("mappings.pkl")
    user_clicks = _load("user_clicks.pkl")

    aid_to_idx     = mappings["article_id_to_emb_index"]
    emb_article_ids = mappings["emb_article_ids"]

    clicked = user_clicks.get(user_id, [])
    indices = [aid_to_idx[aid] for aid in clicked if aid in aid_to_idx]

    if not indices:
        return _fallback(top_n)

    user_profile = embeddings[indices].mean(axis=0, keepdims=True)
    scores       = (embeddings @ user_profile.T).flatten()

    scores[indices] = -np.inf
    top = np.argpartition(scores, -top_n)[-top_n:]
    top = top[np.argsort(scores[top])[::-1]]
    return [int(emb_article_ids[i]) for i in top]


def recommend_similarity(user_id: int, top_n: int = 5) -> list:
    item_sim    = _load("item_similarity.pkl")
    mappings    = _load("mappings.pkl")
    user_clicks = _load("user_clicks.pkl")

    # sim_article_idx/sim_idx_article : sous-ensemble des articles les plus cliqués
    # Fallback sur article_idx pour compatibilité avec les anciens artefacts
    article_idx = mappings.get("sim_article_idx", mappings["article_idx"])
    idx_article = mappings.get("sim_idx_article", mappings["idx_article"])

    clicked = user_clicks.get(user_id, [])
    indices = [article_idx[aid] for aid in clicked if aid in article_idx]

    if not indices:
        return _fallback(top_n)

    scores = item_sim[indices].sum(axis=0)
    return _top_n(scores, idx_article, indices, top_n)


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
                "body": json.dumps({"error": f"Modèle inconnu : {model_name}. Valeurs : {list(MODELS)}"}),
            }

        recommendations = MODELS[model_name](user_id)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "user_id":         user_id,
                "model":           model_name,
                "recommendations": recommendations,
            }),
        }

    except KeyError as e:
        return {"statusCode": 400, "body": json.dumps({"error": f"Paramètre manquant : {e}"})}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
