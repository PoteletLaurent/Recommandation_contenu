"""
streamlit_app.py
-----------------
Interface de gestion du système de recommandation.
- Liste les utilisateurs connus
- Permet de choisir le modèle de recommandation
- Appelle AWS Lambda via boto3
- Affiche les 5 articles recommandés

Variables d'environnement à définir (ou fichier .env) :
    AWS_REGION        – ex: "eu-west-3"
    LAMBDA_FUNCTION   – nom de la fonction Lambda
    S3_BUCKET         – pour afficher les métadonnées articles (optionnel)
"""

import json
import os
import boto3
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
AWS_REGION      = os.environ.get("AWS_REGION", "eu-west-3")
LAMBDA_FUNCTION = os.environ.get("LAMBDA_FUNCTION", "recommandation-handler")
DATA_DIR        = os.environ.get("DATA_DIR", os.path.expanduser("~/news-portal-user-interactions-by-globocom"))

MODEL_OPTIONS = {
    "Similarité item-based (meilleur rappel)": "similarity",
    "Embeddings content-based (PCA)":          "embeddings",
    "ALS – Collaborative Filtering":           "als",
}

# ─────────────────────────────────────────────
# Chargement des données locales (metadata articles + liste users)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    articles = pd.read_csv(
        os.path.join(DATA_DIR, "articles_metadata.csv"),
        usecols=["article_id", "category_id", "words_count"],
    )
    clicks = pd.read_csv(
        os.path.join(DATA_DIR, "clicks_sample.csv"),
        usecols=["user_id", "click_article_id"],
    )
    user_ids = sorted(clicks["user_id"].unique().tolist())
    return articles, clicks, user_ids


@st.cache_resource
def get_lambda_client():
    return boto3.client("lambda", region_name=AWS_REGION)


# ─────────────────────────────────────────────
# Appel Lambda
# ─────────────────────────────────────────────
def call_lambda(user_id: int, model: str) -> list:
    client  = get_lambda_client()
    payload = {"user_id": user_id, "model": model}

    response = client.invoke(
        FunctionName   = LAMBDA_FUNCTION,
        InvocationType = "RequestResponse",
        Payload        = json.dumps(payload).encode(),
    )

    result = json.loads(response["Payload"].read())
    body   = json.loads(result.get("body", "{}"))

    if result.get("statusCode") != 200:
        raise RuntimeError(body.get("error", "Erreur inconnue"))

    return body["recommendations"]


# ─────────────────────────────────────────────
# Interface Streamlit
# ─────────────────────────────────────────────
st.set_page_config(page_title="Recommandation d'articles", layout="wide")

st.title("Système de recommandation d'articles")
st.caption("MVP – Moteur de recommandation via AWS Lambda")

# Chargement
with st.spinner("Chargement des données..."):
    try:
        articles_df, clicks_df, user_ids = load_data()
        data_loaded = True
    except Exception as e:
        st.error(f"Impossible de charger les données locales : {e}")
        data_loaded = False

if data_loaded:
    st.sidebar.header("Paramètres")

    # ── Sélection utilisateur ──────────────────
    user_id = st.sidebar.selectbox(
        "Utilisateur",
        options=user_ids,
        format_func=lambda x: f"User {x}",
    )

    # ── Sélection modèle ──────────────────────
    model_label = st.sidebar.radio(
        "Modèle de recommandation",
        options=list(MODEL_OPTIONS.keys()),
    )
    model_key = MODEL_OPTIONS[model_label]

    # ── Bouton ────────────────────────────────
    run = st.sidebar.button("Obtenir les recommandations", type="primary")

    # ── Historique de l'utilisateur ───────────
    st.subheader(f"Historique de l'utilisateur {user_id}")
    user_history = clicks_df[clicks_df["user_id"] == user_id]["click_article_id"].tolist()

    if user_history:
        history_info = articles_df[articles_df["article_id"].isin(user_history)][
            ["article_id", "category_id", "words_count"]
        ].drop_duplicates()
        st.dataframe(history_info, use_container_width=True)
    else:
        st.info("Aucun historique trouvé pour cet utilisateur (cold start).")

    # ── Recommandations ───────────────────────
    if run:
        st.subheader(f"5 articles recommandés – modèle : {model_label}")

        with st.spinner("Appel de la fonction Lambda..."):
            try:
                reco_ids = call_lambda(user_id, model_key)

                reco_info = articles_df[articles_df["article_id"].isin(reco_ids)][
                    ["article_id", "category_id", "words_count"]
                ].set_index("article_id").reindex(reco_ids).reset_index()

                # Catégories de l'historique utilisateur (pour l'indicateur)
                user_categories = set(
                    articles_df[articles_df["article_id"].isin(user_history)]["category_id"].tolist()
                )

                # Affichage en cartes
                cols = st.columns(5)
                for i, (col, row) in enumerate(zip(cols, reco_info.itertuples())):
                    with col:
                        same_cat = row.category_id in user_categories
                        st.metric(label=f"#{i+1}", value=f"Article {row.article_id}")
                        st.caption(
                            f"Catégorie : {row.category_id}\n"
                            f"Mots      : {row.words_count}"
                        )
                        if same_cat:
                            st.success("Même catégorie que votre historique")

                st.success(f"Recommandations générées par le modèle **{model_label}**.")

            except Exception as e:
                st.error(f"Erreur lors de l'appel Lambda : {e}")
                st.info(
                    "Vérifiez que :\n"
                    "- La variable d'environnement `LAMBDA_FUNCTION` est définie\n"
                    "- Vos credentials AWS sont configurés (`aws configure`)\n"
                    "- La fonction Lambda est déployée et accessible"
                )

    # ── Comparaison des modèles ────────────────
    with st.expander("Comparaison des modèles"):
        st.table(
            pd.DataFrame({
                "Modèle":       ["ALS", "Embeddings (PCA)", "Similarité item-based"],
                "Type":         ["Collaborative filtering", "Content-based", "Item-based"],
                "Rappel@10":    [0.11, 0.21, 0.34],
                "F1-Score":     [0.03, 0.06, 0.07],
                "Cold start ?": ["Non", "Oui (contenu)", "Non"],
            })
        )
