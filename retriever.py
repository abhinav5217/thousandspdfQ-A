import chromadb
from chromadb.config import Settings

# Import the same embedding function used by ingestion.
from embedding_model import get_embedding

# Connect to ChromaDB.
client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db"
    )
)

collection = client.get_or_create_collection(
    name="rag_db"
)

def retrieve(query, top_k=10):
    """Find relevant documents using OpenAI embeddings."""

    # Generate the question embedding using the same model.
    query_embedding = get_embedding(query)

    # Avoid requesting more results than the collection contains.
    total_documents = collection.count()

    if total_documents == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total_documents)
    )

    # Return the matching document text.
    if not results["documents"]:
        return []

    return results["documents"][0]