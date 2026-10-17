
import re
import chromadb
from rank_bm25 import BM25Okapi
from embedding_model import get_embedding

# Use the same database path as ingest.py.
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="rag_db")


def tokenize(text):
    """Convert text to lowercase word tokens for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def retrieve(query, top_k=10, session_id=None):
    """
    Hybrid retrieval using:
    1. Vector Search
    2. BM25 keyword search
    3. Reciprocal Rank Fusion (RRF)

    Both search methods are restricted to one session.
    """

    # Never perform an unfiltered search.
    if not session_id:
        return []

    where_filter = {"session_id": session_id}

    # Get only documents owned by this session.
    stored = collection.get(
        where=where_filter,
        include=["documents", "metadatas"],
    )

    ids = stored.get("ids") or []
    documents = stored.get("documents") or []

    # Ignore empty documents.
    pairs = [
        (doc_id, doc)
        for doc_id, doc in zip(ids, documents)
        if doc and doc.strip()
    ]

    if not pairs:
        return []

    ids = [pair[0] for pair in pairs]
    documents = [pair[1] for pair in pairs]

    candidate_k = min(top_k * 3, len(documents))

    # STEP 1: Vector Search, filtered by session.
    query_embedding = get_embedding(query)

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        where=where_filter,
    )

    vector_ids = (vector_results.get("ids") or [[]])[0]

    # STEP 2: BM25 keyword search on this session's documents only.
    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = sorted(
        range(len(documents)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:candidate_k]

    bm25_ids = [ids[i] for i in bm25_indices]

    # STEP 3: Combine rankings using Reciprocal Rank Fusion.
    rrf_scores = {}
    rrf_constant = 60

    for rank, doc_id in enumerate(vector_ids, start=1):
        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            + 1 / (rrf_constant + rank)
        )

    for rank, doc_id in enumerate(bm25_ids, start=1):
        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            + 1 / (rrf_constant + rank)
        )

    # Return only the highest-ranked chunks.
    id_to_document = dict(zip(ids, documents))
    ranked_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )

    return [
        id_to_document[doc_id]
        for doc_id in ranked_ids[:top_k]
        if doc_id in id_to_document
    ]
