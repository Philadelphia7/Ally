from pathlib import Path

from app.document_loaders import LoadedPage
from app.ingestion import IngestionService
from app.vector_index import LocalVectorIndex


class FakeLoader:
    def load_directory(self, directory: Path):
        assert directory.name == "data"
        return [
            LoadedPage(source="guide.pdf", page=1, text="alpha beta gamma delta"),
            LoadedPage(source="guide.pdf", page=2, text="nutrition and diabetes care"),
        ]


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]


def test_ingestion_builds_and_persists_index(tmp_path):
    index = LocalVectorIndex(tmp_path / "index.json")
    service = IngestionService(
        loader=FakeLoader(),
        embedder=FakeEmbedder(),
        index=index,
        chunk_size=12,
        chunk_overlap=2,
    )

    result = service.ingest(tmp_path / "data")

    assert result.document_count == 1
    assert result.page_count == 2
    assert result.chunk_count >= 2
    assert index.exists() is True

    loaded = LocalVectorIndex(tmp_path / "index.json")
    loaded.load()
    assert len(loaded.records) == result.chunk_count

