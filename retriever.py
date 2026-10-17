
import re
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

# Use the same embedding model as the ingestion pipeline.
from embedding_model import get_embedding


# --------------------------------------------------
# CHROMA DB
# --------------------------------------------------

client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db"
    )
)

collection = client.get_or_create_collection(
    name="rag_db"
)


# --------------------------------------------------
# TEXT TOKENIZATION FOR BM25
# --------------------------------------------------

def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


# --------------------------------------------------
# HYBRID RETRIEVAL
# --------------------------------------------------

def retrieve(query, top_k=10):
    """
    Combine vector similarity search and BM25 keyword search.
    Use Reciprocal Rank Fusion (RRF) to merge both rankings.
    """

    # Get all indexed chunks from ChromaDB.
    stored = collection.get(
        include=["documents", "metadatas"]
    )

    ids = stored.get("ids") or []
    documents = stored.get("documents") or []

    # Return an empty list when no chunks are indexed.
    if not ids or not documents:
        return []

    # Keep only valid documents and their matching IDs.
    pairs = [
        (doc_id, doc)
        for doc_id, doc in zip(ids, documents)
        if doc and doc.strip()
    ]

    if not pairs:
        return []

    ids = [pair[0] for pair in pairs]
    documents = [pair[1] for pair in pairs]

    # ------------------------------------------------
    # STEP 1: VECTOR SEARCH
    # ------------------------------------------------

    query_embedding = get_embedding(query)

    candidate_k = min(top_k * 3, len(documents))

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k
    )

    vector_ids = (
        vector_results.get("ids") or [[]]
    )[0]

    # ------------------------------------------------
    # STEP 2: BM25 KEYWORD SEARCH
    # ------------------------------------------------

    tokenized_corpus = [
        tokenize(doc) for doc in documents
    ]

    # BM25 ranks chunks based on keyword relevance.
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    # Sort chunks by BM25 score, highest first.
    bm25_indices = sorted(
        range(len(documents)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:candidate_k]

    bm25_ids = [
        ids[i] for i in bm25_indices
    ]

    # ------------------------------------------------
    # STEP 3: RECIPROCAL RANK FUSION (RRF)
    # ------------------------------------------------

    # RRF combines rankings without comparing raw scores.
    rrf_scores = {}
    id_to_document = dict(zip(ids, documents))
    rrf_constant = 60

    # Add contributions from vector search.
    for rank, doc_id in enumerate(vector_ids, start=1):
        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            + 1 / (rrf_constant + rank)
        )

    # Add contributions from BM25 keyword search.
    for rank, doc_id in enumerate(bm25_ids, start=1):
        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            + 1 / (rrf_constant + rank)
        )

    # Sort combined results by their fused ranking.
    ranked_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    # Return the top-ranked text chunks.
    return [
        id_to_document[doc_id]
        for doc_id in ranked_ids[:top_k]
        if doc_id in id_to_document
    ]
