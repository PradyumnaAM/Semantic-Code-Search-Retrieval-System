from backend.embeddings import embed_question
from backend.vector_store import search_chunks
from backend.reranker import rerank_results

def evaluate_reranking(test_cases: list, k: int = 3):
    correct = 0

    for test in test_cases:
        question = test["question"]
        expected = test["expected_function"]

        embedded_question = embed_question(question)

        results = search_chunks(embedded_question, 20)

        reranked_chunks = rerank_results(question, results, k)
        
        names = []
        for chunk in reranked_chunks:
            names.append(chunk["metadata"]["name"])

        print(f"\nQuestion: {question}")
        print(f"Expected: {expected}")
        print(f"Retrieved: {names}")

        if expected in names:
            correct += 1
            print("Result: PASS")
        else:
            print("Result: FAIL")

    total = len(test_cases)
    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"\nHit@{k}: {accuracy:.1f}%")
    print(f"Correct: {correct}/{total}")

    return accuracy


test_cases = [
{
    "question": "Which function splits long chunks into smaller pieces?",
    "expected_function": "split_long_chunk"
},
{
    "question": "Which function generates embeddings for code chunks?",
    "expected_function": "embed_chunks"
},
{
    "question": "Which function loads repository files?",
    "expected_function": "load_repository"
},
{
    "question": "Which function stores chunks in the vector database?",
    "expected_function": "store_chunks"
},
{
    "question": "Which function builds context for the LLM?",
    "expected_function": "build_context"
}
]

if __name__ == "__main__":
    evaluate_reranking(test_cases, k=3)
    