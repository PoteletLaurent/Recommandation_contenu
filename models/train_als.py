"""
train_als.py
------------
Entraîne uniquement le modèle ALS (collaborative filtering) et sauvegarde
als_model.pkl. Permet de réentraîner ALS sans relancer les autres modèles.

Usage :
    python models/train_als.py --s3-bucket BUCKET --data-dir DIR
    python models/train_als.py --data-dir DIR --no-upload    # local seulement
    python models/train_als.py --data-dir DIR --evaluate     # + Recall@10
"""

import argparse

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from common import (
    build_als_mappings,
    build_user_clicks,
    load_dataset,
    save_artifact,
    split_clicks,
    upload_to_s3,
)

ARTIFACT = "als_model.pkl"


def train(train_df, factors: int = 50, iterations: int = 20,
          regularization: float = 0.1):
    """Entraîne ALS et retourne (artefacts ALS, mappings ALS)."""
    maps = build_als_mappings(train_df)
    rows = train_df["user_id"].map(maps["user_idx"]).values
    cols = train_df["click_article_id"].map(maps["article_idx"]).values

    sparse = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(maps["user_ids"]), len(maps["article_ids"])),
    )

    print(f"\nEntraînement ALS "
          f"({len(maps['user_ids']):,} users × {len(maps['article_ids']):,} articles)...")
    model = AlternatingLeastSquares(
        factors=factors, iterations=iterations,
        regularization=regularization, use_gpu=False,
    )
    model.fit(sparse)

    artifacts = {
        "user_factors": model.user_factors,
        "item_factors": model.item_factors,
    }
    return artifacts, maps


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--data-dir",
                        default="../news-portal-user-interactions-by-globocom")
    parser.add_argument("--factors",        type=int,   default=50)
    parser.add_argument("--iterations",     type=int,   default=20)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--no-upload", action="store_true",
                        help="Ne pas uploader sur S3 même si --s3-bucket est fourni.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Calcule Recall@10 sur le test set après entraînement.")
    args = parser.parse_args()

    data = load_dataset(args.data_dir)
    train_df, test_df = split_clicks(data["clicks_df"])

    artifacts, maps = train(
        train_df,
        factors=args.factors,
        iterations=args.iterations,
        regularization=args.regularization,
    )

    print("\nSauvegarde de l'artefact...")
    save_artifact(artifacts, ARTIFACT)

    if args.s3_bucket and not args.no_upload:
        print(f"\nUpload vers s3://{args.s3_bucket}/models/ ...")
        upload_to_s3(args.s3_bucket, ARTIFACT)

    if args.evaluate:
        from evaluate import evaluate_als
        user_clicks = build_user_clicks(train_df)
        print("\nÉvaluation Recall@10...")
        evaluate_als(artifacts, maps, user_clicks, test_df)

    print("\nTerminé.")


if __name__ == "__main__":
    _cli()
