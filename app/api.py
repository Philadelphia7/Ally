import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.azure_clients import AzureOpenAIClient, AzureSpeechClient
from app.config import Settings, get_settings
from app.document_loaders import DocumentLoader
from app.ingestion import IngestionService
from app.models import (
    AnswerResponse,
    AskRequest,
    HealthResponse,
    IngestResponse,
    VoiceAnswerResponse,
)
from app.rag import RAGService
from app.tools import build_default_registry
from app.vector_index import LocalVectorIndex


def create_app(
    settings: Settings | None = None,
    rag_service: RAGService | None = None,
    ingestion_service: IngestionService | None = None,
    speech_client: AzureSpeechClient | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title=settings.app_name)

    def get_rag_service() -> RAGService:
        nonlocal rag_service
        if rag_service is None:
            openai_client = AzureOpenAIClient(settings)
            rag_service = RAGService(
                index=LocalVectorIndex(settings.index_path),
                embedder=openai_client,
                chat_client=openai_client,
                tools=build_default_registry(settings.medication_database_url),
                top_k=settings.retrieval_top_k,
            )
        return rag_service

    def get_ingestion_service() -> IngestionService:
        nonlocal ingestion_service
        if ingestion_service is None:
            openai_client = AzureOpenAIClient(settings)
            ingestion_service = IngestionService(
                loader=DocumentLoader(settings),
                embedder=openai_client,
                index=LocalVectorIndex(settings.index_path),
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        return ingestion_service

    def get_speech_client() -> AzureSpeechClient:
        nonlocal speech_client
        if speech_client is None:
            speech_client = AzureSpeechClient(settings)
        return speech_client

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            azure_openai_configured=settings.azure_openai_configured,
            document_intelligence_configured=settings.document_intelligence_configured,
            speech_configured=settings.speech_configured,
            index_exists=settings.index_path.exists(),
        )

    @app.get("/tools")
    def tools() -> list[dict]:
        return build_default_registry(settings.medication_database_url).schemas()

    @app.post("/ingest", response_model=IngestResponse)
    def ingest() -> IngestResponse:
        try:
            result = get_ingestion_service().ingest(settings.data_dir)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return IngestResponse(
            document_count=result.document_count,
            page_count=result.page_count,
            chunk_count=result.chunk_count,
            index_path=str(result.index_path),
        )

    @app.post("/ask", response_model=AnswerResponse)
    def ask(request: AskRequest) -> AnswerResponse:
        try:
            return get_rag_service().answer(request.question)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/voice/ask", response_model=VoiceAnswerResponse)
    async def voice_ask(audio: UploadFile = File(...)) -> VoiceAnswerResponse:
        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix) as audio_file:
            audio_file.write(await audio.read())
            audio_file.flush()
            try:
                transcript = get_speech_client().transcribe_file(Path(audio_file.name))
                answer = get_rag_service().answer(transcript)
                audio_bytes = get_speech_client().synthesize(answer.answer)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        return VoiceAnswerResponse(
            question=answer.question,
            transcript=transcript,
            answer=answer.answer,
            citations=answer.citations,
            tool_results=answer.tool_results,
            audio_base64=base64.b64encode(audio_bytes).decode("ascii"),
            audio_content_type="audio/wav",
        )

    return app

