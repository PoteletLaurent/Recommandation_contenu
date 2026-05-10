"""
evaluate.py
-----------
Évalue Recall@10 sur le jeu de test pour 1 ou plusieurs modèles.
Réutilisable depuis les scripts d'entraînement (option --evaluate).

Usage :
    python models/evaluate.py --data-dir DIR --model all
    python models/evaluate.py --data-dir DIR --model als
    python models/evaluate.py --data-dir DIR --model similarity --s3-bucket BUCKET
        # télécharge depuis S3 si l'artefact local est absent
"""

import argparse

import numpy as np

from common import (
    build_embeddings_mappings,
    build_user_clicks,
    load_artifact,
    load_dataset,
    recall_at_k,
    split_clicks,
)

TOP_N_EVAL = 10


def _ground_truth(test_df) -> dict:
    return test_df.groupby("user_id")["click_article_id"].apply(set).to_dict()


def evaluate_als(als, mappings, user_clicks, test_df, top_n: int = TOP_N_EVAL) -> float:
    user_idx    = mappings["user_idx"]
    article_idx = mappings["article_idx"]
    idx_article = mappings["idx_article"]
    truth = _ground_truth(test_df)

    recalls = []
    for uid, true_set in truth.items():
        if uid not in user_idx:
            continue
        u_vec  = als["user_factors"][user_idx[uid]]
        scores = als["item_factors"] @ u_vec
        seen = [article_idx[a] for a in user_clicks.get(uid, []) if a in article_idx]
        if seen:
            scores = scores.copy()
            scores[seen] = -np.inf
        top = set(idx_article[i] for i in np.argpartition(scores, -top_n)[-top_n:])
        recalls.append(recall_at_k(top, true_set))

    score = float(np.mean(recalls)) if recalls else 0.0
    print(f"  ALS             Recall@{top_n} = {score:.4f}  (n={len(recalls):,} users)")
    return score


def evaluate_embeddings(embeddings_pca, emb_ids, aid_to_idx, user_clicks,
                        test_df, top_n: int = TOP_N_EVAL) -> float:
    truth = _ground_truth(test_df)
    recalls = []
    for uid, true_set in truth.items():
        clicked = user_clicks.get(uid, [])
        idxs = [aid_to_idx[a] for a in clicked if a in aid_to_idx]
        if not idxs:
            continue
        profile = embeddings_pca[idxs].mean(axis=0, keepdims=True)
        scores = (embeddings_pca @ profile.T).flatten()
        scores = scores.copy()
        scores[idxs] = -np.inf
        top = set(int(emb_ids[i]) for i in np.argpartition(scores, -top_n)[-top_n:])
        recalls.append(recall_at_k(top, true_set))

    score = float(np.mean(recalls)) if recalls else 0.0
    print(f"  Embeddings PCA  Recall@{top_n} = {score:.4f}  (n={len(recalls):,} users)")
    return score


def evaluate_similarity(item_sim, sim_article_idx, sim_idx_article,
                        user_clicks, test_df, top_n: int = TOP_N_EVAL) -> float:
    truth = _ground_truth(test_df)
    recalls = []
    for uid, true_set in truth.items():
        clicked = user_clicks.get(uid, [])
        idxs = [sim_article_idx[a] for a in clicked if a in sim_article_idx]
        if not idxs:
            continue
        scores = item_sim[idxs].sum(axis=0)
        scores = scores.copy()
        scores[idxs] = -np.inf
        top = set(sim_idx_article[i] for i in np.argpartition(scores, -top_n)[-top_n:])
        recalls.append(recall_at_k(top, true_set))

    score = float(np.mean(recalls)) if recalls else 0.0
    print(f"  Similarité      Recall@{top_n} = {score:.4f}  (n={len(recalls):,} users)")
    return score


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",
                        default="../news-portal-user-interactions-by-globocom")
    parser.add_argument("--model", choices=["als", "embeddings", "similarity", "all"],
                        default="all")
    parser.add_argument("--s3-bucket", default=None,
                        help="Si fourni, télécharge depuis S3 les artefacts manquants en local.")
    args = parser.parse_args()

    data = load_dataset(args.data_dir)
    train_df, test_df = split_clicks(data["clicks_df"])
    user_clicks = build_user_clicks(train_df)

    print("\nÉvaluation Recall@10 sur le jeu de test...")

    if args.model in ("als", "all"):
        als      = load_artifact("als_model.pkl", bucket=args.s3_bucket)
        mappings = load_artifact("mappings.pkl",  bucket=args.s3_bucket)
        evaluate_als(als, mappings, user_clicks, test_df)

    if args.model in ("embeddings", "all"):
        embeddings_pca = load_artifact("embeddings_pca.pkl", bucket=args.s3_bucket)
        emb_maps = build_embeddings_mappings(data["article_ids_emb"])
        evaluate_embeddings(
            embeddings_pca,
            emb_maps["emb_article_ids"],
            emb_maps["article_id_to_emb_index"],
            user_clicks, test_df,
        )

    if args.model in ("similarity", "all"):
        item_sim = load_artifact("item_similarity.pkl", bucket=args.s3_bucket)
        mappings = load_artifact("mappings.pkl",        bucket=args.s3_bucket)
        evaluate_similarity(
            item_sim,
            mappings["sim_article_idx"], mappings["sim_idx_article"],
            user_clicks, test_df,
        )

    print("\nTerminé.")


if __name__ == "__main__":
    _cli()
