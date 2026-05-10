"""
train_models.py
---------------
Orchestrateur : prépare les artefacts partagés, entraîne les 3 modèles
puis évalue Recall@10 sur le test set. Conserve l'interface CLI historique.

Pour réentraîner / évaluer un seul modèle, utiliser plutôt :
    models/prepare_data.py
    models/train_als.py
    models/train_embeddings.py
    models/train_similarity.py
    models/evaluate.py

Usage :
    python models/train_models.py --s3-bucket BUCKET --data-dir DIR
    python models/train_models.py --s3-bucket BUCKET --data-dir DIR --max-sim-articles 10000
"""

import argparse

import evaluate as eval_mod
import prepare_data
import train_als
import train_embeddings
import train_similarity
from common import (
    build_embeddings_mappings,
    build_user_clicks,
    load_dataset,
    save_artifact,
    split_clicks,
    upload_to_s3,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--data-dir",
                        default="../news-portal-user-interactions-by-globocom")
    parser.add_argument("--pca-components",   type=int, default=64)
    parser.add_argument("--max-sim-articles", type=int, default=10000)
    args = parser.parse_args()

    # 1. Données + split unique (déterministe → cohérence inter-modèles)
    data = load_dataset(args.data_dir)
    train_df, test_df = split_clicks(data["clicks_df"])
    user_clicks = build_user_clicks(train_df)

    # 2. Artefacts partagés (mappings + user_clicks) — délégué à prepare_data
    shared = prepare_data.run(
        args.data_dir,
        s3_bucket=args.s3_bucket,
        max_sim_articles=args.max_sim_articles,
    )
    mappings = shared["mappings"]

    # 3. Entraînement des 3 modèles
    als_artifacts, _ = train_als.train(train_df)
    save_artifact(als_artifacts, train_als.ARTIFACT)
    upload_to_s3(args.s3_bucket, train_als.ARTIFACT)

    embeddings_pca = train_embeddings.train(data["embeddings_matrix"], args.pca_components)
    save_artifact(embeddings_pca, train_embeddings.ARTIFACT)
    upload_to_s3(args.s3_bucket, train_embeddings.ARTIFACT)

    sim_matrix, _, _ = train_similarity.train(train_df, args.max_sim_articles)
    save_artifact(sim_matrix, train_similarity.ARTIFACT)
    upload_to_s3(args.s3_bucket, train_similarity.ARTIFACT)

    # 4. Évaluation
    print("\nÉvaluation Recall@10 sur le jeu de test...")
    eval_mod.evaluate_als(als_artifacts, mappings, user_clicks, test_df)
    emb_maps = build_embeddings_mappings(data["article_ids_emb"])
    eval_mod.evaluate_embeddings(
        embeddings_pca,
        emb_maps["emb_article_ids"],
        emb_maps["article_id_to_emb_index"],
        user_clicks, test_df,
    )
    eval_mod.evaluate_similarity(
        sim_matrix,
        mappings["sim_article_idx"], mappings["sim_idx_article"],
        user_clicks, test_df,
    )

    print("\nTerminé.")


if __name__ == "__main__":
    main()
