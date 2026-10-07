import os
import uuid
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


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
# EMBEDDING MODEL
# --------------------------------------------------

embed_model = SentenceTransformer(
    "all-mpnet-base-v2"
)


# --------------------------------------------------
# LOAD PDF
# --------------------------------------------------

def load_pdf(file_path):

    reader = PdfReader(file_path)

    pages = []

    for i, page in enumerate(reader.pages):

        text = page.extract_text() or ""

        pages.append(
            (text, i + 1)
        )

    return pages


# --------------------------------------------------
# CHUNK TEXT
# --------------------------------------------------

def chunk_text(
    text,
    chunk_size=500,
    overlap=100
):

    chunks = []

    step = chunk_size - overlap

    for i in range(
        0,
        len(text),
        step
    ):

        chunk = text[
            i:i + chunk_size
        ]

        if chunk.strip():

            chunks.append(
                chunk
            )

    return chunks


# --------------------------------------------------
# INGEST ONE UPLOADED PDF
# --------------------------------------------------

def ingest_uploaded_pdf(
    file_path,
    filename
):

    pages = load_pdf(
        file_path
    )

    total_chunks = 0

    for text, page_no in pages:

        chunks = chunk_text(
            text
        )

        for chunk_index, chunk in enumerate(chunks):

            embedding = embed_model.encode(
                chunk
            ).tolist()

            chunk_id = (
                f"{filename}_"
                f"{page_no}_"
                f"{chunk_index}_"
                f"{uuid.uuid4().hex}"
            )

            collection.add(

                ids=[
                    chunk_id
                ],

                embeddings=[
                    embedding
                ],

                documents=[
                    chunk
                ],

                metadatas=[
                    {
                        "file": filename,
                        "page": page_no
                    }
                ]
            )

            total_chunks += 1

    return total_chunks


# --------------------------------------------------
# INGEST ALL PDFs FROM DATA FOLDER
# --------------------------------------------------

def ingest(folder="data"):

    total_chunks = 0

    for file in os.listdir(folder):

        if file.lower().endswith(".pdf"):

            file_path = os.path.join(
                folder,
                file
            )

            chunks_added = ingest_uploaded_pdf(
                file_path,
                file
            )

            total_chunks += chunks_added

            print(
                f"✅ {file}: "
                f"{chunks_added} chunks"
            )

    print(
        f"\n✅ Total chunks: {total_chunks}"
    )


# --------------------------------------------------
# RUN DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":

    ingest()