from sentence_transformers import CrossEncoder


# --------------------------------------------------
# CROSS ENCODER
# --------------------------------------------------

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# --------------------------------------------------
# RERANK
# --------------------------------------------------

def rerank(
    query,
    docs,
    top_k=5
):

    if not docs:

        return []


    # Create query-document pairs
    pairs = [
        (query, doc)
        for doc in docs
    ]


    # Calculate relevance scores
    scores = model.predict(
        pairs
    )


    # Combine documents + scores
    ranked = sorted(

        zip(
            docs,
            scores
        ),

        key=lambda x: x[1],

        reverse=True
    )


    # Return top documents
    return [
        doc
        for doc, score
        in ranked[:top_k]
    ]