"""Tests for corpus loading, chunking, and semantic retrieval."""

import pytest

import knowledge
from knowledge import build_knowledge_base, chunk_documents, load_corpus, retrieve


def test_load_corpus_and_chunk_documents_from_temp_fixture(tmp_path) -> None:
    """Corpus parsing reads simple headers and creates unique chunk IDs."""
    folder = tmp_path / "cover_cropping"
    folder.mkdir()
    (folder / "soil.txt").write_text(
        "\n".join(
            [
                "INTERVENTION: cover_cropping",
                "METRIC: soil_organic_carbon",
                "SOURCE: Test Source",
                "SOURCE_URL: https://example.com/source",
                "REGION: test",
                "---",
                "Cover crops add biomass and protect soil.",
            ],
        ),
        encoding="utf-8",
    )
    (folder / "water.txt").write_text(
        "\n".join(
            [
                "INTERVENTION: cover_cropping",
                "METRIC: water_infiltration",
                "SOURCE: Test Source",
                "SOURCE_URL: https://example.com/source",
                "REGION: test",
                "---",
                "Cover crops improve infiltration.\n\nThey also reduce runoff.",
            ],
        ),
        encoding="utf-8",
    )

    documents = load_corpus(str(tmp_path))
    chunks = chunk_documents(documents)

    assert len(documents) == 2
    assert {document["intervention"] for document in documents} == {"cover_cropping"}
    assert len(chunks) == 2
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)


def test_load_corpus_rejects_missing_required_header(tmp_path) -> None:
    """A missing required header fails with the file name in the error."""
    folder = tmp_path / "cover_cropping"
    folder.mkdir()
    path = folder / "broken.txt"
    path.write_text(
        "\n".join(
            [
                "INTERVENTION: cover_cropping",
                "SOURCE: Test Source",
                "SOURCE_URL: https://example.com/source",
                "REGION: test",
                "---",
                "Missing the metric header.",
            ],
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="broken.txt"):
        load_corpus(str(tmp_path))


@pytest.mark.slow
def test_retrieve_ranks_matching_synthetic_chunk_first(tmp_path, monkeypatch) -> None:
    """A tiny Chroma collection returns the closest semantic topic first."""
    monkeypatch.setattr(knowledge, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(knowledge, "CHROMA_COLLECTION_NAME", "test_knowledge")

    chunks = [
        {
            "chunk_id": "cover_chunk_0",
            "intervention": "cover_cropping",
            "metric": "soil_organic_carbon",
            "source": "Synthetic",
            "source_url": "https://example.com/cover",
            "region": "test",
            "text": "Cover crops add residue, roots, and soil carbon in cropland systems.",
        },
        {
            "chunk_id": "windbreak_chunk_0",
            "intervention": "windbreaks_buffer_strips",
            "metric": "erosion_risk",
            "source": "Synthetic",
            "source_url": "https://example.com/wind",
            "region": "test",
            "text": "Windbreaks slow wind, reduce soil erosion, and shelter field edges.",
        },
        {
            "chunk_id": "riparian_chunk_0",
            "intervention": "riparian_buffers",
            "metric": "water_quality",
            "source": "Synthetic",
            "source_url": "https://example.com/riparian",
            "region": "test",
            "text": "Riparian buffers filter nutrients and sediment before runoff reaches streams.",
        },
    ]

    build_knowledge_base(chunks, reset=True)

    results = retrieve("soil carbon from cover crops", top_k=3, similarity_cutoff=0.0)
    unrelated_results = retrieve(
        "invoice software payment reminders",
        top_k=3,
        similarity_cutoff=0.95,
    )

    assert results[0]["chunk_id"] == "cover_chunk_0"
    assert unrelated_results == []
