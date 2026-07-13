from app.chunking import TextChunk
from app.vector_index import LocalVectorIndex


def test_vector_index_persists_and_returns_nearest_chunks(tmp_path):
    index_path = tmp_path / "index.json"
    index = LocalVectorIndex(index_path)
    chunks = [
        TextChunk(chunk_id="a:1:0", source="a.pdf", page=1, text="diabetes care"),
        TextChunk(chunk_id="b:1:0", source="b.pdf", page=1, text="nutrition guidance"),
    ]

    index.add(chunks[0], [1.0, 0.0])
    index.add(chunks[1], [0.0, 1.0])
    index.save()

    loaded = LocalVectorIndex(index_path)
    loaded.load()
    results = loaded.search([0.8, 0.2], top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "a:1:0"
    assert results[0].score > 0.9


def test_vector_index_reports_missing_index(tmp_path):
    index = LocalVectorIndex(tmp_path / "missing.json")

    assert index.exists() is False
    try:
        index.load()
    except FileNotFoundError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")

