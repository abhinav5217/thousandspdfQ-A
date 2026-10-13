
from sentence_transformers import SentenceTransformer

# Load the embedding model in one shared place.
# Other modules can import this same model instance.
embed_model = SentenceTransformer(
    "all-mpnet-base-v2",
    device="cpu"
)
