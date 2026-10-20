from backend.reranker import rerank_results
from backend.embeddings import embed_question
from backend.vector_store import search_chunks


question = "Which function builds context?"
embedding = embed_question(question)
results = search_chunks(embedding, k=20)

ranked = rerank_results(question, results, top_k=3)

for item in ranked:
    print(item["metadata"]["name"], item["score"])