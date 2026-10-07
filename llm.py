import os

from openai import OpenAI
from dotenv import load_dotenv


# --------------------------------------------------
# OPENAI
# --------------------------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv(
        "OPENAI_API_KEY"
    )
)


# --------------------------------------------------
# GENERATE ANSWER
# --------------------------------------------------

def generate_answer(
    query,
    context
):

    prompt = f"""
Answer using ONLY the context below.

Rules:

1. Be accurate.
2. Do not guess.
3. Do not use outside knowledge.
4. If the answer is not present in the context,
   say:
   "I could not find the answer in the uploaded documents."

Context:
{context}

Question:
{query}
"""


    response = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    return response.choices[0].message.content