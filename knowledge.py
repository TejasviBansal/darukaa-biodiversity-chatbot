"""Corpus loading, chunking, ChromaDB ingestion, and semantic retrieval."""

import logging
import os
import time
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CORPUS_ROOT_DIR,
    EMBEDDING_MODEL_NAME,
    MAX_FINAL_CHUNKS,
    RETRIEVAL_SIMILARITY_CUTOFF,
    RETRIEVAL_TOP_K,
)

logger = logging.getLogger(__name__)

REQUIRED_HEADER_FIELDS = ("INTERVENTION", "METRIC", "SOURCE", "SOURCE_URL", "REGION")


# --- 1. Corpus loading ---


def load_corpus(corpus_root: str = CORPUS_ROOT_DIR) -> list[dict[str, str]]:
    """
    Walk corpus_root for .txt files, parse the header + body, return documents.

    Each returned dict has: {"doc_id", "intervention", "metric", "source",
    "source_url", "region", "text"}. Raises ValueError naming the file if a
    required header field is missing.
    """
    root = Path(corpus_root)
    documents: list[dict[str, str]] = []

    for path in sorted(root.rglob("*.txt")):
        raw_text = path.read_text(encoding="utf-8").strip()
        if "---" not in raw_text:
            raise ValueError(f"Missing header separator in corpus file: {path}")

        header_text, body = raw_text.split("---", 1)
        headers = _parse_headers(header_text, path)
        missing = [field for field in REQUIRED_HEADER_FIELDS if field not in headers]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Missing required header field(s) in {path}: {missing_text}")

        doc_id = path.relative_to(root).with_suffix("").as_posix().replace("/", "__")
        documents.append(
            {
                "doc_id": doc_id,
                "intervention": headers["INTERVENTION"],
                "metric": headers["METRIC"],
                "source": headers["SOURCE"],
                "source_url": headers["SOURCE_URL"],
                "region": headers["REGION"],
                "text": body.strip(),
            },
        )

    return documents


def _parse_headers(header_text: str, path: Path) -> dict[str, str]:
    """Parse simple KEY: value headers from one corpus file."""
    headers: dict[str, str] = {}
    for line in header_text.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"Invalid header line in {path}: {line}")
        key, value = line.split(":", 1)
        headers[key.strip()] = value.strip()
    return headers


# --- 2. Chunking ---


def chunk_documents(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Split longer documents into paragraph chunks while preserving metadata.

    Most Phase 2 corpus files are short enough to be a single chunk. If a
    document's word count exceeds about 300 words, split on paragraph breaks
    and never mid-sentence. Each chunk has a unique "chunk_id".
    """
    chunks: list[dict[str, str]] = []
    for document in documents:
        chunk_texts = _split_text_on_paragraphs(document["text"])
        for index, text in enumerate(chunk_texts):
            chunk = {key: value for key, value in document.items() if key != "text"}
            chunk["chunk_id"] = f"{document['doc_id']}_chunk_{index}"
            chunk["text"] = text
            chunks.append(chunk)
    return chunks


def _split_text_on_paragraphs(text: str, max_words: int = 300) -> list[str]:
    """Return one or more paragraph-based chunks without splitting sentences."""
    if len(text.split()) <= max_words:
        return [text]

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph_words = len(paragraph.split())
        if current and current_words + paragraph_words > max_words:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_words = paragraph_words
        else:
            current.append(paragraph)
            current_words += paragraph_words

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# --- 3. Ingestion ---


def build_knowledge_base(chunks: list[dict[str, str]], reset: bool = False) -> int:
    """
    Embed chunks and store them in a persistent cosine-space ChromaDB collection.

    If reset=True, delete and recreate the collection first. Returns the count
    of chunks ingested. Logs model load time and ingestion time.
    """
    start_time = time.perf_counter()
    model = _get_model()
    logger.info("Embedding model ready in %.2f seconds", time.perf_counter() - start_time)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except (ValueError, chromadb.errors.NotFoundError):
            pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True).tolist()
    metadatas = [_metadata_for_chroma(chunk) for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.info("Ingested %s chunks in %.2f seconds", len(chunks), time.perf_counter() - start_time)
    return len(chunks)


def _metadata_for_chroma(chunk: dict[str, str]) -> dict[str, str]:
    """Return Chroma metadata without the large text body."""
    return {
        "intervention": chunk["intervention"],
        "metric": chunk["metric"],
        "source": chunk["source"],
        "source_url": chunk["source_url"],
        "region": chunk["region"],
    }


# --- 4. Retrieval ---


_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load and cache the embedding model once per process."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    similarity_cutoff: float = RETRIEVAL_SIMILARITY_CUTOFF,
    max_results: int = MAX_FINAL_CHUNKS,
) -> list[dict[str, str | float]]:
    """
    Embed a query and return the most relevant knowledge chunks.

    Chroma returns cosine distance because the collection uses cosine space.
    Convert each distance to similarity with: similarity = 1 - distance.
    Results are filtered by similarity_cutoff, capped to max_results, and
    sorted by descending similarity.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(CHROMA_COLLECTION_NAME)
    query_embedding = _get_model().encode([query], convert_to_numpy=True)[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = _format_retrieval_results(results, similarity_cutoff)
    retrieved.sort(key=lambda result: float(result["similarity_score"]), reverse=True)
    return retrieved[:max_results]


def _format_retrieval_results(
    results: dict[str, Any],
    similarity_cutoff: float,
) -> list[dict[str, str | float]]:
    """Convert raw Chroma query output into app-level result dictionaries."""
    output: list[dict[str, str | float]] = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    result_rows = zip(ids, documents, metadatas, distances, strict=True)
    for chunk_id, text, metadata, distance in result_rows:
        similarity = 1 - float(distance)
        if similarity < similarity_cutoff:
            continue
        output.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "intervention": metadata["intervention"],
                "metric": metadata["metric"],
                "source": metadata["source"],
                "source_url": metadata["source_url"],
                "region": metadata["region"],
                "similarity_score": similarity,
            },
        )

    return output


if os.getenv("DARUKAA_LOG_KNOWLEDGE_IMPORT") == "1":
    logger.debug("Knowledge module loaded with corpus root %s", CORPUS_ROOT_DIR)
