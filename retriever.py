import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
# Reuse the same embedding model used by ingest.py.
from embedding_model import embed_model


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

embed_model = SentenceTransformer(
    "all-mpnet-base-v2"
)


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
# RETRIEVE
# --------------------------------------------------

def retrieve(
    query,
    top_k=10
):

    # Convert question into embedding
    query_embedding = embed_model.encode(
        query
    ).tolist()


    # Search ChromaDB
    results = collection.query(

        query_embeddings=[
            query_embedding
        ],

        n_results=top_k
    )


    # No documents
    if not results["documents"]:

        return []


    # Return document text
    return results["documents"][0]