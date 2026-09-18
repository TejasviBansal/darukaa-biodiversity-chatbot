# Darukaa.Earth AI Biodiversity Intelligence Chatbot

## Phase 1

Phase 1 is a network-free foundation for classifying ecological metrics, detecting
simple cross-variable biodiversity interactions, and storing short-lived chat
session state in memory.

## Phase 2

Phase 2 adds a curated intervention corpus, ChromaDB ingestion, and semantic
retrieval. Corpus files are plain `.txt` documents with simple headers followed
by `---` and the body text:

```text
INTERVENTION: cover_cropping
METRIC: soil_organic_carbon
SOURCE: FAO Soil Organic Cover in Conservation Agriculture
SOURCE_URL: https://www.fao.org/conservation-agriculture/in-practice/soil-organic-cover/en/
REGION: general
---
Body text...
```

Build the local knowledge base with:

```bash
python scripts/build_knowledge_base.py
```

Retrieval embeds the query, fetches top-k Chroma candidates, converts cosine
distance to similarity, applies the configured similarity cutoff, and caps the
final result count. The default similarity cutoff is `0.42`, tuned from Phase 2
test queries to keep about 3-5 useful chunks after the final cap. With
`all-mpnet-base-v2`, short scenario queries compared against longer explanatory
passages often score lower than same-length sentence comparisons, so a cutoff
near `0.5` was too strict for this corpus.

## Install

```bash
pip install -r requirements.txt
```

## Test

```bash
pytest
pytest -m "not slow"
ruff check .
mypy .
```

## Files

- `config.py`: environment-driven settings and all scientific threshold bands.
- `classifier.py`: classifies metric values into sourced threshold bands.
- `interactions.py`: detects ecological interaction rules from classifications.
- `knowledge.py`: loads the corpus, chunks documents, ingests ChromaDB, and retrieves chunks.
- `memory.py`: thread-safe in-memory session and history store.
- `corpus/`: curated Phase 2 intervention knowledge files.
- `scripts/build_knowledge_base.py`: rebuilds the persistent ChromaDB collection.
- `tests/`: focused pytest coverage for Phase 1 behavior.
