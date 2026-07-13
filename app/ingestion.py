from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.chunking import TextChunk, chunk_text
from app.document_loaders import LoadedPage
from app.vector_index import LocalVectorIndex


class PageLoader(Protocol):
    def load_directory(self, directory: Path) -> list[LoadedPage]:
        ...


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class IngestionResult:
    document_count: int
    page_count: int
    chunk_count: int
    index_path: Path


class IngestionService:
    def __init__(
        self,
        loader: PageLoader,
        embedder: Embedder,
        index: LocalVectorIndex,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self.loader = loader
        self.embedder = embedder
        self.index = index
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, data_dir: Path) -> IngestionResult:
        pages = self.loader.load_directory(data_dir)
        chunks: list[TextChunk] = []
        for page in pages:
            chunks.extend(
                chunk_text(
                    page.text,
                    source=page.source,
                    page=page.page,
                    chunk_size=self.chunk_size,
                    overlap=self.chunk_overlap,
                )
            )

        embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError("embedder returned a different number of embeddings than chunks")

        self.index.clear()
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.index.add(chunk, embedding)
        self.index.save()

        return IngestionResult(
            document_count=len({page.source for page in pages}),
            page_count=len(pages),
            chunk_count=len(chunks),
            index_path=self.index.path,
        )

