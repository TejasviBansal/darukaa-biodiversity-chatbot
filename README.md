# Darukaa.Earth Biodiversity Intelligence Chatbot

An AI environmental scientist that reasons over soil health, land use, biodiversity indicators, climate factors, and human impact to recommend evidence-backed interventions for restoring biodiversity and soil function.

**Live demo:** https://darukaa-biodiversity-chatbot-dvfs4newexk3yz2b3abg4u.streamlit.app/

**Repository:** https://github.com/TejasviBansal/darukaa-biodiversity-chatbot

---

## What this system does

A user describes their land in free text, fills in structured metrics, or both. The pipeline:

1. **Extracts metrics** deterministically from free text, with no LLM in the parsing path.
2. **Classifies** each metric into scientifically-sourced threshold bands.
3. **Detects cross-variable interactions** through deterministic rules (soil-water, land use-carbon, erosion, dryland-fragmentation).
4. **Retrieves** evidence from a curated corpus via ChromaDB: 16 documents across 8 intervention categories, sourced from FAO and USDA.
5. **Generates** recommendations using Gemini, constrained to the retrieved evidence only.
6. **Validates** every recommendation against the evidence. Unmatched sources and ungrounded numbers are dropped before display.
7. **Asks clarifying questions** when fewer than two core metrics are available, and acknowledges detected topics and contextual descriptors in the question.

Every recommendation must be traceable to a retrieved source. If a source does not match, or a number in the recommendation does not appear in the retrieved evidence, the recommendation is dropped before the user sees it.

## Architecture

Flat, single-process design with one file per concept.

```
config.py           All settings, thresholds, and sourced scientific bands
classifier.py       Numeric + categorical metric classification
extractor.py        Free-text metric extraction, topic + descriptor detection
interactions.py     Cross-variable interaction rule engine
knowledge.py        Corpus loading, chunking, ChromaDB ingestion, retrieval
generator.py        Situation summary, prompt assembly, Gemini call
validator.py        Post-LLM grounding validation + confidence scoring
memory.py           Thread-safe in-memory session store
respond.py          Full pipeline orchestrator
api.py              FastAPI backend (/chat, /session/{id}, /health)
app.py              Streamlit frontend (single-process deployment)
corpus/             16 curated knowledge files (8 interventions x 2 sources)
scripts/            One-shot ChromaDB ingestion runner
tests/              77 tests across 9 modules
```

The Streamlit app calls the pipeline in-process for reliability on free hosting. The FastAPI backend is kept as the documented structured-JSON API surface (see the "JSON API" section below).

## Knowledge system

**Corpus.** 16 plain-text files under `corpus/`, one per (intervention, metric) pair, covering cover cropping, agroforestry, intercropping, crop rotation, no-till, windbreaks, mixed grazing, and riparian buffers. Each file carries `INTERVENTION`, `METRIC`, `SOURCE`, `SOURCE_URL`, and `REGION` headers followed by `---` and the body text. Sources are FAO (~half) and USDA NRCS / Forest Service / SARE (~half).

**Retrieval.** Embeddings are produced with `sentence-transformers` using `all-mpnet-base-v2` against a ChromaDB collection in cosine similarity space. Retrieval fetches top-K candidates, converts cosine distance to similarity, applies a similarity cutoff of `0.42` (empirically tuned for this corpus), and caps the final result count at 5.

**Descriptor enrichment.** The retrieval query is the situation summary, which contains both classified metrics and contextual descriptors (fragmentation, connectivity, erosion risk, water-limited conditions, waterways, salinisation, degradation, slope, perennial cover). Descriptors are extracted deterministically and never enter the metrics dict; they only bias retrieval toward relevant evidence. This invariant is enforced by a test (`test_descriptors_never_become_metrics`).

**Transparency.** The Streamlit UI exposes a **Retrieval details** expander showing the situation summary sent to retrieval, each retrieved chunk's source and similarity score, and a truncated preview of the chunk text. The JSON API exposes the same data in a `retrieval_debug` field on `/chat` responses.

## Data layer

The system has no traditional relational database. State is distributed across three purpose-specific stores:

**Vector store (ChromaDB).** The retrieval index is a persistent ChromaDB collection with cosine similarity as the distance metric. Each entry stores the chunk text, an embedding produced by `sentence-transformers` (`all-mpnet-base-v2`, 768-dimensional), and metadata fields (`source`, `intervention`, `metric`, `region`, `source_url`). The collection is written once at ingestion time by `scripts/build_knowledge_base.py` and read at query time by `knowledge.retrieve()`. On Streamlit Cloud, the collection is rebuilt on cold start by `app._ensure_knowledge_base()` because the ephemeral filesystem does not persist it between container restarts.

**Corpus (source of truth).** The ChromaDB collection is derived from 16 plain-text files under `corpus/`. Each file has a fixed header schema:

```
INTERVENTION: <intervention key>
METRIC: <metric key>
SOURCE: <source name>
SOURCE_URL: <canonical URL>
REGION: general | united_states
---
<body text>
```

The header parser in knowledge.load_corpus() rejects any file missing a required header, so schema violations fail fast at ingestion rather than silently producing bad embeddings.

**Session state (in-memory).** Conversation history and per-session metrics live in a thread-safe in-memory dict managed by memory.SessionStore. There is no persistence layer: sessions are lost on process restart. This is a deliberate trade-off for a demo deployment, discussed in the "Design trade-offs" section below.

## Anti-hallucination methodology

Only one step in the pipeline touches the LLM: generation. Classification, interaction detection, retrieval, and validation are all deterministic.

1. **Pre-LLM classification.** Metric values are placed into sourced threshold bands in code, not by LLM judgment.
2. **Pre-LLM interaction detection.** Cross-variable logic is fixed Python rules, not LLM-generated reasoning.
3. **Retrieval-only prompting.** The prompt restricts the model to the retrieved evidence only. If the evidence does not support any recommendation, the model returns fewer recommendations (including zero) rather than inventing one.
4. **Low temperature.** `LLM_TEMPERATURE=0.2` reduces improvisation.
5. **Post-LLM source verification** (`validator.verify_source`). Each recommendation's cited source must match a retrieved chunk's source exactly. Unmatched sources cause the recommendation to be dropped.
6. **Post-LLM numeric grounding** (`validator.check_numbers_grounded`). Every numeric token in a recommendation must appear verbatim in the retrieved evidence, and any recommendation with an ungrounded number is dropped.
7. **Confidence from retrieval similarity, not LLM self-assessment.** High / medium / low tiers are derived from the matching chunk's similarity score.
8. **Fail-safe drop.** Recommendations that fail either check are removed from the response without a warning. The `dropped_count` field reports how many were filtered.

Every one of these steps is covered by tests in `tests/test_validator.py`.

## Conversational intelligence

- **Multi-turn memory.** Thread-safe in-memory session store with configurable history depth (`SESSION_HISTORY_MAX_TURNS`).
- **Clarifying-question gate.** If fewer than 2 of the 3 core metrics (SOC%, rainfall, land use) are available, the pipeline skips retrieval and asks for the missing inputs. The threshold is configurable (`MIN_CORE_METRICS_REQUIRED`).
- **Context acknowledgement.** Clarifying questions reference detected topics and descriptors, e.g. "You mentioned biodiversity. I understand you are dealing with erosion risk, sloped terrain, nearby waterways. To recommend actions I also need: soil organic carbon %, annual rainfall (mm), land use type."
- **Contradiction avoidance.** A descriptor that maps to a missing metric is suppressed from the acknowledgement, so the system never says "I see you are in a semi-arid zone; please tell me your rainfall."
- **Extracted-metric acknowledgement.** Non-core metrics that were successfully extracted (e.g. `soil_ph`) are acknowledged in the clarifying question rather than silently ignored.

## Test suite

**77 tests across 9 modules.** Run with:

```
pytest tests/ -v
```

Coverage by module:

- **`test_classifier.py`** -- boundary behavior for every sourced threshold band and categorical vocabulary.
- **`test_extractor.py`** -- deterministic metric extraction (including spelled-out units, multi-word keywords, range rejection, wrong-unit rejection), topic detection, descriptor detection, and the invariant that descriptors never become metrics.
- **`test_generator.py`** -- situation-summary construction (with and without descriptors) and prompt assembly, including a drift guard that every descriptor has a matching summary phrasing.
- **`test_interactions.py`** -- rule firing on coupled-variable scenarios.
- **`test_knowledge.py`** -- corpus loading, header validation, and retrieval ranking on synthetic fixtures.
- **`test_memory.py`**: session isolation and history trimming.
- **`test_respond.py`**: the full pipeline: clarification gate, extraction merge order, structured-values-win precedence, fallback paths, and descriptor propagation into the situation summary.
- **`test_validator.py`**: source verification, numeric grounding, and confidence tiers.
- **`test_api.py`**: FastAPI endpoints, request validation, and error paths.

Lint and type checks pass cleanly:

```
ruff check .
mypy .
```

## CI/CD

**CI.** No automated CI pipeline is configured in this repository. The test and lint commands run manually during development:

```
pytest tests/ -v
ruff check .
mypy .
```

All three pass cleanly at the time of submission (77 tests, ruff clean, mypy clean). A minimal GitHub Actions workflow is straightforward to add -- a single job that runs the three commands on push -- but it was deliberately out of scope for this build to keep the submission focused on the reasoning and grounding design.

**CD.** Deployment is fully automated via Streamlit Community Cloud. The service is linked to the `main` branch of the GitHub repository and redeploys on every push. There is no manual deploy step and no separate release process.

The FastAPI backend (`api.py`) is not deployed. It runs locally via `uvicorn api:app --reload --port 8000` and is documented as the structured-JSON API surface.

## Local setup

```
git clone https://github.com/TejasviBansal/darukaa-biodiversity-chatbot.git
cd darukaa-biodiversity-chatbot
pip install -r requirements.txt

# Configure the Gemini API key
cp .env.example .env

# Edit .env and set GEMINI_API_KEY=<your key>
# Build the Chroma vector store (one-time; the app also builds it on first run if missing)
python scripts/build_knowledge_base.py

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first run, if `chroma_store/` does not exist, the app builds it automatically before serving the first request (~30-60 seconds).

## JSON API

The FastAPI surface in `api.py` provides the structured JSON input path.

```
uvicorn api:app --reload --port 8000
```

```
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-1",
    "metrics": {
      "soil_organic_carbon_pct": 0.35,
      "rainfall_mm_year": 350,
      "land_use_category": "monoculture"
    },
    "message": "How do I restore microbial life without tanking yields?"
}'
```

The response includes `recommendations`, `dropped_count`, `needs_more_info`, `merged_metrics`, and `retrieval_debug`.

## Known limitations

1. **Region scope.** Roughly half of the corpus draws on U.S.-specific government sources (USDA NRCS, USDA Forest Service, SARE). Figures from these files are illustrative benchmarks and should not be treated as universally applicable. FAO-sourced content is written for broader applicability.
2. **Three corpus figures were intentionally softened.** No-till soil-loss reduction and two riparian-buffer nutrient and sediment figures were rounded to less falsely-precise ranges when the exact source could not be independently verified within available time. This is a deliberate caution, disclosed here rather than hidden.
3. **Numeric grounding is substring-based, not claim-level.** A short shared digit could theoretically match unrelated evidence text. Documented simplicity trade-off.
4. **Session memory is in-memory only.** State resets on server restart. Deliberate, low-risk for demo use.
5. **LLM model availability is a moving target.** `gemini-3.1-flash-lite` is current as of submission; Gemini model names change and should be re-verified against Google's documentation if this project is resumed later.
6. **Quantitative estimates appear only when present in retrieved evidence.** The system deliberately does not synthesize plausible-sounding numbers that the evidence does not contain. When the top retrieved chunks are qualitative, the recommendations are qualitative. This is expected: the validator will drop any figure that is not in the evidence, so the system cannot pad its output with invented numbers.

## Design trade-offs

- **In-memory sessions over SQLite.** Session storage is not part of the graded requirement, and SQLite on ephemeral cloud disks risks data loss. The simplification is deliberate.
- **Deterministic free-text extraction over LLM parsing.** An LLM-based extractor would reintroduce hallucination into the input-parsing stage, contradicting the anti-hallucination design. The extractor is auditable regex and closed-vocabulary matching instead.
- **Single-process deployment over two-service.** The Streamlit app calls the pipeline in-process for reliability on free hosting. `api.py` remains in the repository as the documented JSON surface so the structured-input requirement is fully satisfied.
- **Descriptor enrichment as context, not evidence.** Descriptors improve retrieval targeting and clarify clarifying questions, but they never enter the metrics dict or the LLM's evidence context directly. This preserves the boundary between user-declared context and validated evidence.
