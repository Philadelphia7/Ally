from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class Citation(BaseModel):
    source: str
    page: int | None
    score: float
    text: str


class ToolResult(BaseModel):
    name: str
    result: dict[str, Any]


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    tool_results: list[dict[str, Any]]


class IngestResponse(BaseModel):
    document_count: int
    page_count: int
    chunk_count: int
    index_path: str


class VoiceAnswerResponse(AnswerResponse):
    transcript: str
    audio_base64: str
    audio_content_type: str = "audio/wav"


class HealthResponse(BaseModel):
    status: str
    azure_openai_configured: bool
    document_intelligence_configured: bool
    speech_configured: bool
    index_exists: bool

