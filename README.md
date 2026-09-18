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

## Phase 3

Phase 3 adds the generation pipeline and post-LLM grounding validator. The LLM
prompt includes the situation summary, retrieved corpus chunks, exact source
names, and strict JSON output instructions. After Gemini responds, the validator
drops any recommendation whose cited source does not exactly match a retrieved
chunk source, or whose numeric claims do not appear in the retrieved evidence.
Surviving recommendations receive a confidence tier from the matching chunk's
retrieval similarity score, not from LLM self-assessment. The numeric grounding
check uses simple substring matching rather than claim-level association, so a
short shared digit could theoretically match unrelated context; this is a known
simplicity tradeoff.

Set `GEMINI_API_KEY` in `.env` before calling `generate_recommendations()` or
`get_response()`. Get a key from Google AI Studio and never commit it. The
default generation model is `gemini-3.1-flash-lite`; Gemini model availability
changes over time, so periodically verify this value against Google's model
documentation: https://ai.google.dev/gemini-api/docs/models.

## Install

```bash
pip install -r requirements.txt
```

For local development, copy `.env.example` to `.env` and fill in real values,
especially `GEMINI_API_KEY` before running anything that calls Gemini. `.env`
is loaded automatically with `python-dotenv`, so you do not need to manually
export variables in your shell.

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
- `generator.py`: builds situation summaries, prompts Gemini, and parses JSON recommendations.
- `knowledge.py`: loads the corpus, chunks documents, ingests ChromaDB, and retrieves chunks.
- `memory.py`: thread-safe in-memory session and history store.
- `respond.py`: runs the full classify, retrieve, generate, validate pipeline.
- `validator.py`: drops ungrounded LLM recommendations and assigns confidence.
- `corpus/`: curated Phase 2 intervention knowledge files.
- `scripts/build_knowledge_base.py`: rebuilds the persistent ChromaDB collection.
- `tests/`: focused pytest coverage for Phase 1 behavior.
