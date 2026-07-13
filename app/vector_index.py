import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from app.chunking import TextChunk


@dataclass(frozen=True)
class SearchResult:
    chunk: TextChunk
    score: float


@dataclass(frozen=True)
class VectorRecord:
    chunk: TextChunk
    embedding: list[float]


class LocalVectorIndex:
    def __init__(self, path: Path):
        self.path = path
        self.records: list[VectorRecord] = []

    def exists(self) -> bool:
        return self.path.exists()

    def clear(self) -> None:
        self.records = []

    def add(self, chunk: TextChunk, embedding: Iterable[float]) -> None:
        vector = [float(value) for value in embedding]
        if not vector:
            raise ValueError("embedding must not be empty")
        self.records.append(VectorRecord(chunk=chunk, embedding=vector))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"chunk": asdict(record.chunk), "embedding": record.embedding}
            for record in self.records
        ]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Vector index is missing: {self.path}")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.records = [
            VectorRecord(
                chunk=TextChunk(**item["chunk"]),
                embedding=[float(value) for value in item["embedding"]],
            )
            for item in payload
        ]

    def search(self, query_embedding: Iterable[float], top_k: int = 5) -> list[SearchResult]:
        query = [float(value) for value in query_embedding]
        scored = [
            SearchResult(chunk=record.chunk, score=_cosine_similarity(query, record.embedding))
            for record in self.records
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    return dot_product / (left_norm * right_norm)

