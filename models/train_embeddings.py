"""
train_embeddings.py
-------------------
Entraîne uniquement le modèle Embeddings + PCA (content-based) et sauvegarde
embeddings_pca.pkl. Indépendant des clics (basé sur le contenu des articles).

Usage :
    python models/train_embeddings.py --s3-bucket BUCKET --data-dir DIR
    python models/train_embeddings.py --data-dir DIR --no-upload
    python models/train_embeddings.py --data-dir DIR --evaluate
    python models/train_embeddings.py --data-dir DIR --pca-components 32
"""

import argparse

import numpy as np
from sklearn.decomposition import PCA

from common import (
    RANDOM_STATE,
    build_embeddings_mappings,
    build_user_clicks,
    load_dataset,
    save_artifact,
    split_clicks,
    upload_to_s3,
)

ARTIFACT = "embeddings_pca.pkl"


def train(embeddings_matrix: np.ndarray, n_components: int = 64) -> np.ndarray:
    print(f"\nPCA sur les embeddings ({embeddings_matrix.shape[1]}D → {n_components}D)...")
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    embeddings_pca = pca.fit_transform(embeddings_matrix).astype(np.float32)
    print(f"  Variance expliquée : {pca.explained_variance_ratio_.sum():.2%}")
    return embeddings_pca


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--data-dir",
                        default="../news-portal-user-interactions-by-globocom")
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--evaluate",  action="store_true",
                        help="Calcule Recall@10 sur le test set après entraînement.")
    args = parser.parse_args()

    data = load_dataset(args.data_dir)
    embeddings_pca = train(data["embeddings_matrix"], args.pca_components)

    print("\nSauvegarde de l'artefact...")
    save_artifact(embeddings_pca, ARTIFACT)

    if args.s3_bucket and not args.no_upload:
        print(f"\nUpload vers s3://{args.s3_bucket}/models/ ...")
        upload_to_s3(args.s3_bucket, ARTIFACT)

    if args.evaluate:
        from evaluate import evaluate_embeddings
        train_df, test_df = split_clicks(data["clicks_df"])
        user_clicks = build_user_clicks(train_df)
        emb_maps = build_embeddings_mappings(data["article_ids_emb"])
        print("\nÉvaluation Recall@10...")
        evaluate_embeddings(
            embeddings_pca,
            emb_maps["emb_article_ids"],
            emb_maps["article_id_to_emb_index"],
            user_clicks,
            test_df,
        )

    print("\nTerminé.")


if __name__ == "__main__":
    _cli()
