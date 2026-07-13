from app.chunking import chunk_text


def test_chunk_text_preserves_metadata_and_overlap():
    chunks = chunk_text(
        text="abcdefghijklmnopqrstuvwxyz",
        source="alpha.pdf",
        page=3,
        chunk_size=10,
        overlap=3,
    )

    assert [chunk.text for chunk in chunks] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]
    assert chunks[0].source == "alpha.pdf"
    assert chunks[0].page == 3
    assert chunks[1].chunk_id == "alpha.pdf:3:1"


def test_chunk_text_rejects_overlap_greater_than_chunk_size():
    try:
        chunk_text("hello", source="a.pdf", page=1, chunk_size=10, overlap=10)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected overlap validation error")

