
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = r"C:\Users\Sheru\Downloads\NLP-20261121T074752Z-1-001\NLP\phishing_papers"
ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "saved_retrieval_artifacts"
)

CHUNKS_PATH = os.path.join(
    ARTIFACT_DIR,
    "retrieval_chunks.csv"
)

EMBEDDINGS_PATH = os.path.join(
    ARTIFACT_DIR,
    "sbert_embeddings.npy"
)

BM25_PATH = os.path.join(
    ARTIFACT_DIR,
    "bm25_index.pkl"
)


# --------------------------------------------------
# 2. LOAD MODELS AND RETRIEVAL ARTIFACTS
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading research retrieval models...")
def load_engine():

    chunks = pd.read_csv(CHUNKS_PATH)

    embeddings = np.load(EMBEDDINGS_PATH)

    with open(BM25_PATH, "rb") as file:
        bm25 = pickle.load(file)

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    )

    return (
        chunks,
        embeddings,
        bm25,
        embedding_model,
        cross_encoder
    )


# --------------------------------------------------
# 3. BM25 RETRIEVAL
# --------------------------------------------------

def bm25_search(query, chunks, bm25, top_k=5):

    scores = bm25.get_scores(
        query.lower().split()
    )

    indices = np.argsort(scores)[::-1][:top_k]

    results = chunks.iloc[indices].copy()

    results["BM25_Score"] = scores[indices]

    return results


# --------------------------------------------------
# 4. SBERT RETRIEVAL
# --------------------------------------------------

def sbert_search(
    query,
    chunks,
    embeddings,
    embedding_model,
    top_k=5
):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    indices = np.argsort(similarities)[::-1][:top_k]

    results = chunks.iloc[indices].copy()

    results["SBERT_Score"] = similarities[indices]

    return results


# --------------------------------------------------
# 5. HYBRID RECIPROCAL RANK FUSION
# --------------------------------------------------

def hybrid_search(
    query,
    chunks,
    bm25,
    embeddings,
    embedding_model,
    candidate_k=20,
    top_k=5,
    rrf_constant=60
):

    bm25_results = bm25_search(
        query, chunks, bm25, candidate_k
    )

    sbert_results = sbert_search(
        query, chunks, embeddings,
        embedding_model, candidate_k
    )

    rrf_scores = {}
    bm25_ranks = {}
    sbert_ranks = {}

    for rank, chunk_id in enumerate(
        bm25_results["Chunk_ID"], start=1
    ):

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (rrf_constant + rank)
        )

        bm25_ranks[chunk_id] = rank

    for rank, chunk_id in enumerate(
        sbert_results["Chunk_ID"], start=1
    ):

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (rrf_constant + rank)
        )

        sbert_ranks[chunk_id] = rank

    ranked_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )[:top_k]

    results = (
        chunks.set_index("Chunk_ID")
        .loc[ranked_ids]
        .reset_index()
    )

    results["RRF_Score"] = [
        rrf_scores[cid] for cid in ranked_ids
    ]

    results["BM25_Rank"] = [
        bm25_ranks.get(cid, np.nan)
        for cid in ranked_ids
    ]

    results["SBERT_Rank"] = [
        sbert_ranks.get(cid, np.nan)
        for cid in ranked_ids
    ]

    return results


# --------------------------------------------------
# 6. CROSS-ENCODER RERANKING
# --------------------------------------------------

def rerank_results(
    query,
    candidate_df,
    cross_encoder,
    top_k=5
):

    if candidate_df.empty:
        return candidate_df.copy()

    pairs = [
        (query, str(text))
        for text in candidate_df["Text"]
    ]

    scores = cross_encoder.predict(pairs)

    results = candidate_df.copy()

    results["CrossEncoder_Score"] = scores

    results = results.sort_values(
        "CrossEncoder_Score",
        ascending=False
    )

    return results.head(top_k).reset_index(drop=True)


# --------------------------------------------------
# 7. MAIN RETRIEVAL CONTROLLER
# --------------------------------------------------

def retrieve(
    query,
    method,
    top_k,
    chunks,
    bm25,
    embeddings,
    embedding_model,
    cross_encoder
):

    if not query or not query.strip():
        raise ValueError("Please enter a research question.")

    if method == "No Retrieval":

        return pd.DataFrame()

    elif method == "BM25":

        return bm25_search(
            query, chunks, bm25, top_k
        )

    elif method == "SBERT":

        return sbert_search(
            query, chunks, embeddings,
            embedding_model, top_k
        )

    elif method == "Hybrid RRF":

        return hybrid_search(
            query,
            chunks,
            bm25,
            embeddings,
            embedding_model,
            candidate_k=20,
            top_k=top_k,
            rrf_constant=60
        )

    elif method == "Hybrid RRF + Cross-Encoder":

        candidates = hybrid_search(
            query,
            chunks,
            bm25,
            embeddings,
            embedding_model,
            candidate_k=20,
            top_k=20,
            rrf_constant=60
        )

        return rerank_results(
            query,
            candidates,
            cross_encoder,
            top_k=top_k
        )

    else:

        raise ValueError("Unknown retrieval configuration.")
