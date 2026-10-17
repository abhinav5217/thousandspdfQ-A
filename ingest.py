
import os
import uuid
import chromadb
from pypdf import PdfReader
from embedding_model import get_embedding

# Use a persistent ChromaDB database.
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="rag_db")


def load_pdf(file_path):
    """Extract text from each PDF page."""
    reader = PdfReader(file_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((text, i + 1))

    return pages


def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks."""
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def ingest_uploaded_pdf(file_path, filename, session_id):
    """Store PDF chunks with their owning browser session ID."""
    pages = load_pdf(file_path)
    total_chunks = 0

    for text, page_no in pages:
        chunks = chunk_text(text)

        for chunk_index, chunk in enumerate(chunks):
            # Generate an embedding for the chunk.
            embedding = get_embedding(chunk)

            # Generate a unique ID for every stored chunk.
            chunk_id = uuid.uuid4().hex

            # Save the session ID so retrieval can filter by session.
            metadata = {
                "file": filename,
                "page": page_no,
                "session_id": session_id,
            }

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[metadata],
            )

            total_chunks += 1

    return total_chunks


def ingest(folder="data", session_id="default"):
    """Optional helper to index PDFs from a local folder."""
    total_chunks = 0

    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder, filename)

            chunks_added = ingest_uploaded_pdf(
                file_path,
                filename,
                session_id,
            )

            total_chunks += chunks_added
            print(f"{filename}: {chunks_added} chunks")

    print(f"Total chunks: {total_chunks}")


if __name__ == "__main__":
    ingest()
