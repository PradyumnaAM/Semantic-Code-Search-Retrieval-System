
from pathlib import Path

from backend.repository import load_repository
from backend.embeddings import embed_chunks
from backend.vector_store import (
    store_chunks,
    get_active_collection_name,
    create_staging_collection,
    activate_collection,
    delete_collection,
)


def index_repository(repo_path):
    repo = Path(repo_path)

    if not repo.is_dir():
        raise ValueError("Invalid repository directory")

    documents = load_repository(repo_path)

    all_embedded_chunks = []
    files_indexed = 0

    for document in documents:
        if Path(document["path"]).suffix != ".py":
            continue

        embedded_document = embed_chunks(document)

        if not embedded_document:
            continue

        all_embedded_chunks.extend(embedded_document)
        files_indexed += 1

    if not all_embedded_chunks:
        raise ValueError("No Python code chunks found to index")

    previous_collection = get_active_collection_name()
    staging_name, staging_collection = (
        create_staging_collection()
    )

    try:
        stored_count = store_chunks(
            all_embedded_chunks,
            collection=staging_collection
        )

        if staging_collection.count() != stored_count:
            raise RuntimeError(
                "Index verification failed: "
                "stored chunk count does not match"
            )

        activate_collection(staging_name)

    except Exception:
        # Do not affect the previously active index.
        try:
            delete_collection(staging_name)
        except Exception:
            pass

        raise

    # Delete the previous collection only after
    # the replacement has been activated.
    if previous_collection != staging_name:
        try:
            delete_collection(previous_collection)
        except Exception:
            # Cleanup failure should not invalidate
            # an otherwise successful index.
            pass

    return {
        "files_indexed": files_indexed,
        "chunks_stored": stored_count
    }
