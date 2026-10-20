from pathlib import Path

from backend.repository import load_repository
from backend.embeddings import embed_chunks


def test_integration():
    print("\n===== REPOMIND DAY 7 INTEGRATION TEST =====")

    # Change this to the repository you want to test
    repo_path = Path(__file__).resolve().parents[1]

    print(f"\nLoading repository: {repo_path}")

    documents = load_repository(repo_path)

    total_chunks = 0
    split_chunks = 0
    errors = []

    for document in documents:
        if not str(document["path"]).endswith(".py"):
            continue

        print(f"\nFILE: {document['path']}")

        embedded_chunks = embed_chunks(document)

        for item in embedded_chunks:
            chunk = item["chunk"]
            embedding = item["embedding"]

            name = chunk["name"]
            start_line = chunk["start_line"]
            end_line = chunk["end_line"]

            print(
                f"  {name}: lines {start_line}-{end_line} "
                f"| embedding dimensions: {len(embedding)}"
            )
            if name == "chunk_python_document":
                print("\n--- CHUNK CONTENT ---")
                print(chunk["content"][:200])
            total_chunks += 1

            # Validate line numbers
            if start_line > end_line:
                errors.append(
                    f"{name}: start_line > end_line"
                )

            # Validate embedding
            if len(embedding) == 0:
                errors.append(
                    f"{name}: empty embedding"
                )

            # Detect possible split chunks
            if chunk["content"].count("\n") > 0:
                split_chunks += 1
            if name == "chunk_python_document":
                print("\n--- CHUNK CONTENT ---")
                print(chunk["content"][:200])
    print("\n===== TEST SUMMARY =====")
    print(f"Total chunks: {total_chunks}")
    print(f"Multiline chunks: {split_chunks}")
    print(f"Errors: {len(errors)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
    else:
        print("\nAll basic integration checks passed!")


if __name__ == "__main__":
    test_integration()