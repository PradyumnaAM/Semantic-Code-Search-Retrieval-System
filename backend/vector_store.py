import chromadb
from backend.embeddings import embed_chunks, embed_question
from backend.retrieval import retrieve_chunks
from chromadb.errors import ChromaError
from pathlib import Path

def get_collection():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name="repomind_code")

    return collection


def store_chunks(embedded_chunks):
    collection = get_collection()

    ids = []
    documents = []
    embeddings = []
    metadata = []

    for index, item in enumerate(embedded_chunks):
        chunk = item["chunk"]
        embedding = item["embedding"]

        chunk_id = f"{chunk['path']}:{chunk['name']}:{index}"
        ids.append(chunk_id)
        
        documents.append(chunk["content"])
        
        embed_list = embedding.tolist()
        embeddings.append(embed_list)

        metadata_content = {
            "name" : chunk["name"],
            "path" : chunk["path"],
            "type" : chunk["type"],
            "start_line" : chunk["start_line"],
            "end_line" : chunk["end_line"]
        }

        metadata.append(metadata_content)
    
    if ids:
        collection.upsert(
            ids = ids,
            documents = documents,
            embeddings = embeddings,
            metadatas = metadata
        )

    return len(ids)

def search_chunks(question_embedding, k=5):
    try:
        collection = get_collection()
        if collection.count() == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        results = collection.query(
            query_embeddings= [question_embedding.tolist()],
            n_results = min(k, collection.count()),
            include = ["documents","metadatas","distances"]
        )

        return results 
    except ChromaError:
        raise 

def reset_storage():
    Chroma = chromadb.PersistentClient(path = "./data/chroma")
    try:
        Chroma.delete_collection(name="repomind_code")
    except ValueError:
        print("Collection does not exist. Creating a new one.")
    return get_collection()

    
