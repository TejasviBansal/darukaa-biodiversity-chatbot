"""Build the local ChromaDB knowledge base from the curated corpus."""

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge import build_knowledge_base, chunk_documents, load_corpus  # noqa: E402


def main() -> None:
    """Load, chunk, and ingest the curated corpus."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    start_time = time.perf_counter()
    documents = load_corpus()
    chunks = chunk_documents(documents)
    ingested_count = build_knowledge_base(chunks, reset=True)
    interventions = sorted({document["intervention"] for document in documents})
    elapsed = time.perf_counter() - start_time

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chunks ingested: {ingested_count}")
    print(f"Interventions represented: {', '.join(interventions)}")
    print(f"Time elapsed: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
