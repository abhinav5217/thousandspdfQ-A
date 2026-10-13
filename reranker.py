
# Use lightweight keyword-based reranking instead of CrossEncoder.
# This avoids loading another local AI model into RAM.

import re


def tokenize(text):
    """Convert text to lowercase words."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def rerank(query, docs, top_k=5):
    """Rank documents by keyword overlap with the question."""

    if not docs:
        return []

    query_words = tokenize(query)

    # Score each document based on matching question words.
    ranked = []

    for doc in docs:
        doc_words = tokenize(doc)

        # Count query words that appear in the document.
        score = len(query_words & doc_words)

        ranked.append((doc, score))

    # Sort from highest score to lowest score.
    ranked.sort(key=lambda item: item[1], reverse=True)

    # Return only the top-ranked documents.
    return [doc for doc, score in ranked[:top_k]]
