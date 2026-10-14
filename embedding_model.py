import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env locally.
# On Render, variables should be set in the Environment dashboard.
load_dotenv()

# Create the OpenAI client using your existing API key.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    """Generate an embedding using OpenAI's hosted API."""

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding