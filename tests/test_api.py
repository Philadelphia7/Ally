from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.models import AnswerResponse, Citation, IngestResponse


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
            index_path="data/index.json",
        )


class FakeSpeechClient:
    def transcribe_file(self, audio_path):
        return "What is diabetes?"

    def synthesize(self, text, audio_format="wav"):
        return f"{audio_format}-audio-bytes".encode()


def test_root_route_returns_project_details():
    app = create_app(
        rag_service=FakeRAGService(),
        ingestion_service=FakeIngestionService(),
        speech_client=FakeSpeechClient(),
    )
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "Ally Healthwise RAG"
    assert payload["status"] == "ok"
    assert payload["docs_url"] == "/docs"
    assert "/ask" in payload["endpoints"]["rag"]
    assert "/speech/synthesize" in payload["endpoints"]["speech"]


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


def test_voice_route_can_return_mp3_audio():
    app = create_app(
        rag_service=FakeRAGService(),
        ingestion_service=FakeIngestionService(),
        speech_client=FakeSpeechClient(),
    )
    client = TestClient(app)

    response = client.post(
        "/voice/ask",
        data={"audio_format": "mp3"},
        files={"audio": ("question.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["audio_content_type"] == "audio/mpeg"
    assert payload["audio_format"] == "mp3"


def test_speech_transcribe_route_returns_transcript():
    app = create_app(
        rag_service=FakeRAGService(),
        ingestion_service=FakeIngestionService(),
        speech_client=FakeSpeechClient(),
    )
    client = TestClient(app)

    response = client.post(
        "/speech/transcribe",
        files={"audio": ("question.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"transcript": "What is diabetes?"}


def test_speech_synthesize_route_returns_audio_metadata():
    app = create_app(
        rag_service=FakeRAGService(),
        ingestion_service=FakeIngestionService(),
        speech_client=FakeSpeechClient(),
    )
    client = TestClient(app)

    response = client.post(
        "/speech/synthesize",
        json={"text": "Take your medicine after food."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["audio_base64"] == "d2F2LWF1ZGlvLWJ5dGVz"
    assert payload["audio_content_type"] == "audio/wav"
    assert payload["audio_format"] == "wav"


def test_speech_synthesize_route_can_return_mp3_audio():
    app = create_app(
        rag_service=FakeRAGService(),
        ingestion_service=FakeIngestionService(),
        speech_client=FakeSpeechClient(),
    )
    client = TestClient(app)

    response = client.post(
        "/speech/synthesize",
        json={"text": "Take your medicine after food.", "audio_format": "mp3"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["audio_base64"] == "bXAzLWF1ZGlvLWJ5dGVz"
    assert payload["audio_content_type"] == "audio/mpeg"
    assert payload["audio_format"] == "mp3"
