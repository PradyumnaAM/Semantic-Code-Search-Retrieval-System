
from pathlib import Path
from uuid import uuid4

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "repomind_code"
ACTIVE_COLLECTION_FILE = CHROMA_PATH / "active_collection.txt"


def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_active_collection_name():
    if ACTIVE_COLLECTION_FILE.exists():
        name = ACTIVE_COLLECTION_FILE.read_text(
            encoding="utf-8"
        ).strip()

        if name:
            return name

    return COLLECTION_NAME


def get_collection():
    client = get_client()

    return client.get_or_create_collection(
        name=get_active_collection_name()
    )


def store_chunks(embedded_chunks, collection=None):
    if collection is None:
        collection = get_collection()

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for index, item in enumerate(embedded_chunks):
        chunk = item["chunk"]
        embedding = item["embedding"]

        chunk_id = (
            f"{chunk['path']}:{chunk['name']}:{index}"
        )

        ids.append(chunk_id)
        documents.append(chunk["content"])
        embeddings.append(embedding.tolist())

        metadatas.append({
            "name": chunk["name"],
            "path": chunk["path"],
            "type": chunk["type"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"]
        })

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    return len(ids)


def search_chunks(question_embedding, k=5):
    collection = get_collection()

    if collection.count() == 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    return collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )


def create_staging_collection():
    client = get_client()
    name = f"repomind_staging_{uuid4().hex}"

    collection = client.create_collection(name=name)

    return name, collection


def activate_collection(name):
    client = get_client()

    # Verify the replacement before switching.
    collection = client.get_collection(name=name)

    if collection.count() == 0:
        raise ValueError("Cannot activate an empty index")

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    temp_file = ACTIVE_COLLECTION_FILE.with_suffix(".tmp")
    temp_file.write_text(name, encoding="utf-8")

    # Atomically replace the active collection pointer.
    temp_file.replace(ACTIVE_COLLECTION_FILE)


def delete_collection(name):
    client = get_client()
    client.delete_collection(name=name)


def reset_storage():
    client = get_client()
    name = get_active_collection_name()

    try:
        client.delete_collection(name=name)
    except ValueError:
        pass

    ACTIVE_COLLECTION_FILE.unlink(missing_ok=True)

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )
