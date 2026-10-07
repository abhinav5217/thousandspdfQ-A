#
from retriever import retrieve
from reranker import rerank
from llm import generate_answer

print("🚀 PRO RAG READY (type exit)\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    retrieved = retrieve(query)
    best_results = rerank(query, retrieved)

    print("\n📄 Sources:")
    for r in best_results:
        print(f"{r['meta']['file']} - Page {r['meta']['page']}")

    answer = generate_answer(query, best_results)

    print("\n🤖 Answer:", answer, "\n")