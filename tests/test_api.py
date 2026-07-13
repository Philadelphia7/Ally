from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.models import AnswerResponse, Citation, IngestResponse, VoiceAnswerResponse


class FakeRAGService:
    def answer(self, question):
        return AnswerResponse(
            question=question,
            answer="Use the guidance.",
            citations=[Citation(source="guide.pdf", page=1, score=0.99, text="Use the guidance.")],
            tool_results=[],
        )


class FakeIngestionService:
    def ingest(self, data_dir: Path):
        return IngestResponse(
            document_count=1,
            page_count=2,
            chunk_count=3,
            index_path=".ally_index/index.json",
        )


class FakeSpeechClient:
    def transcribe_file(self, audio_path):
        return "What is diabetes?"

    def synthesize(self, text):
        return b"audio-bytes"


def test_ask_route_returns_answer():
    app = create_app(
        rag_service=FakeRAGService(),
        ingestion_service=FakeIngestionService(),
        speech_client=FakeSpeechClient(),
    )
    client = TestClient(app)

    response = client.post("/ask", json={"question": "What should I do?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Use the guidance."


def test_ingest_route_returns_counts():
    app = create_app(
        rag_service=FakeRAGService(),
        ingestion_service=FakeIngestionService(),
        speech_client=FakeSpeechClient(),
    )
    client = TestClient(app)

    response = client.post("/ingest")

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 3


def test_voice_route_returns_transcript_and_audio_metadata():
    app = create_app(
        rag_service=FakeRAGService(),
        ingestion_service=FakeIngestionService(),
        speech_client=FakeSpeechClient(),
    )
    client = TestClient(app)

    response = client.post(
        "/voice/ask",
        files={"audio": ("question.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["transcript"] == "What is diabetes?"
    assert payload["audio_content_type"] == "audio/wav"

