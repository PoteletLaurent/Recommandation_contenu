"""
prepare_data.py
---------------
Construit les artefacts partagés par tous les modèles :
    - mappings.pkl     (ALS + embeddings + similarité réunis dans un seul fichier
                        pour préserver le contrat avec lambda/handler.py)
    - user_clicks.pkl  (historique utilisateur pré-calculé depuis train_df)

Doit être lancé avant (ou indépendamment de) train_als.py / train_embeddings.py /
train_similarity.py si l'on veut que la Lambda dispose d'artefacts à jour.

Usage :
    python models/prepare_data.py --s3-bucket BUCKET --data-dir DIR
    python models/prepare_data.py --data-dir DIR              # sans upload S3
"""

import argparse

from common import (
    build_als_mappings,
    build_embeddings_mappings,
    build_similarity_mappings,
    build_user_clicks,
    load_dataset,
    save_artifact,
    split_clicks,
    upload_to_s3,
)


def run(data_dir: str, s3_bucket: str = None, max_sim_articles: int = 10000) -> dict:
    data = load_dataset(data_dir)
    train_df, _ = split_clicks(data["clicks_df"])

    als_maps                              = build_als_mappings(train_df)
    sim_article_idx, sim_idx_article, _   = build_similarity_mappings(train_df, max_sim_articles)
    emb_maps                              = build_embeddings_mappings(data["article_ids_emb"])

    mappings = {
        **als_maps,
        **emb_maps,
        "sim_article_idx": sim_article_idx,
        "sim_idx_article": sim_idx_article,
    }
    user_clicks = build_user_clicks(train_df)

    print("\nSauvegarde des artefacts partagés...")
    save_artifact(mappings,    "mappings.pkl")
    save_artifact(user_clicks, "user_clicks.pkl")

    if s3_bucket:
        print(f"\nUpload vers s3://{s3_bucket}/models/ ...")
        upload_to_s3(s3_bucket, "mappings.pkl")
        upload_to_s3(s3_bucket, "user_clicks.pkl")

    return {"mappings": mappings, "user_clicks": user_clicks,
            "train_df": train_df, "data": data}


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket", default=None,
                        help="Si fourni, uploade mappings.pkl et user_clicks.pkl sur S3.")
    parser.add_argument("--data-dir",
                        default="../news-portal-user-interactions-by-globocom")
    parser.add_argument("--max-sim-articles", type=int, default=10000,
                        help="Top N articles pour la matrice de similarité (défaut : 10000).")
    args = parser.parse_args()
    run(args.data_dir, args.s3_bucket, args.max_sim_articles)
    print("\nTerminé.")


if __name__ == "__main__":
    _cli()
