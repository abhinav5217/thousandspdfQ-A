

from sentence_transformers import SentenceTransformer

# Use a smaller embedding model to reduce RAM usage.
embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

