from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(MODEL_NAME)

def rerank_results(question: str, results: dict, top_k: int = 3) -> list:
    if not results.get("documents") or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    pairs=[]
    ranked_results = []

    for document, metadata in zip(documents, metadatas):
        code_text = (
            f"Function: {metadata['name']}\n"
            f"File: {metadata['path']}\n"
            f"Code:\n{document}"
        )
        pairs.append((question, code_text))
    
    scores = reranker.predict(pairs)
    

    for document, metadata, score in zip(documents, metadatas, scores):
        ranked = {
            "document": document,
            "metadata": metadata,
            "score" : float(score)
        }
        
        ranked_results.append(ranked)
    
    ranked_results.sort(key=lambda item: item["score"], reverse = True)

    return ranked_results[:top_k]
        
