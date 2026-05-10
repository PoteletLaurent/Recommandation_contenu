"""
train_similarity.py
-------------------
Entraîne uniquement le modèle de Similarité item-based et sauvegarde
item_similarity.pkl. Limité aux MAX_SIM_ARTICLES articles les plus cliqués
pour éviter une matrice N² × 4 octets trop volumineuse.

Usage :
    python models/train_similarity.py --s3-bucket BUCKET --data-dir DIR
    python models/train_similarity.py --data-dir DIR --max-sim-articles 5000
    python models/train_similarity.py --data-dir DIR --evaluate
"""

import argparse

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from common import (
    build_similarity_mappings,
    build_user_clicks,
    load_dataset,
    save_artifact,
    split_clicks,
    upload_to_s3,
)

ARTIFACT = "item_similarity.pkl"


def train(train_df, max_articles: int = 10000):
    """Construit la matrice cosinus item-item sur le top N articles cliqués.

    Retourne (matrice_similarité, sim_article_idx, sim_idx_article).
    """
    sim_article_idx, sim_idx_article, top_aids = build_similarity_mappings(
        train_df, max_articles
    )

    train_sim   = train_df[train_df["click_article_id"].isin(top_aids)]
    user_lookup = {uid: i for i, uid in enumerate(sorted(train_sim["user_id"].unique()))}
    sim_rows    = train_sim["user_id"].map(user_lookup).values
    sim_cols    = train_sim["click_article_id"].map(sim_article_idx).values

    print(f"\nCalcul de la matrice de similarité (top {max_articles:,} articles)...")
    sparse = csr_matrix(
        (np.ones(len(sim_rows)), (sim_rows, sim_cols)),
        shape=(len(user_lookup), len(top_aids)),
    )
    matrix = cosine_similarity(sparse.T).astype(np.float32)
    print(f"  Matrice similarité : {matrix.shape}  ({matrix.nbytes/1e6:.0f} Mo)")

    return matrix, sim_article_idx, sim_idx_article


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--data-dir",
                        default="../news-portal-user-interactions-by-globocom")
    parser.add_argument("--max-sim-articles", type=int, default=10000)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--evaluate",  action="store_true",
                        help="Calcule Recall@10 sur le test set après entraînement.")
    args = parser.parse_args()

    data = load_dataset(args.data_dir)
    train_df, test_df = split_clicks(data["clicks_df"])

    matrix, sim_article_idx, sim_idx_article = train(train_df, args.max_sim_articles)

    print("\nSauvegarde de l'artefact...")
    save_artifact(matrix, ARTIFACT)

    if args.s3_bucket and not args.no_upload:
        print(f"\nUpload vers s3://{args.s3_bucket}/models/ ...")
        upload_to_s3(args.s3_bucket, ARTIFACT)

    if args.evaluate:
        from evaluate import evaluate_similarity
        user_clicks = build_user_clicks(train_df)
        print("\nÉvaluation Recall@10...")
        evaluate_similarity(matrix, sim_article_idx, sim_idx_article,
                            user_clicks, test_df)

    print("\nTerminé.")


if __name__ == "__main__":
    _cli()
