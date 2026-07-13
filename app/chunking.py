from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    source: str
    page: int | None
    text: str


def chunk_text(
    text: str,
    source: str,
    page: int | None = None,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    cleaned = " ".join(text.split()) if len(text) > chunk_size else text.strip()
    if not cleaned:
        return []

    chunks: list[TextChunk] = []
    step = chunk_size - overlap
    for chunk_number, start in enumerate(range(0, len(cleaned), step)):
        chunk = cleaned[start : start + chunk_size].strip()
        if not chunk:
            continue
        chunks.append(
            TextChunk(
                chunk_id=f"{source}:{page or 0}:{chunk_number}",
                source=source,
                page=page,
                text=chunk,
            )
        )
        if start + chunk_size >= len(cleaned):
            break
    return chunks

