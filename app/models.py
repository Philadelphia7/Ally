from typing import Any, Literal

from pydantic import BaseModel, Field

AudioFormat = Literal["wav", "mp3"]


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Plain-language health or medication question to answer.",
        examples=["What are recommended interventions for chronic disease prevention?"],
    )


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
    audio_content_type: str
    audio_format: AudioFormat


class TranscriptionResponse(BaseModel):
    transcript: str = Field(description="Recognized text from the uploaded audio.")


class SynthesisRequest(BaseModel):
    text: str = Field(
        min_length=1,
        description="Text to convert to speech.",
        examples=["Take your medicine after food."],
    )
    audio_format: AudioFormat = Field(
        default="wav",
        description="Audio export format. Use wav for RIFF PCM or mp3 for compressed MPEG audio.",
        examples=["wav"],
    )


class SynthesisResponse(BaseModel):
    audio_base64: str = Field(description="Base64-encoded audio bytes.")
    audio_content_type: str
    audio_format: AudioFormat


class HealthResponse(BaseModel):
    status: str
    azure_openai_configured: bool
    document_intelligence_configured: bool
    speech_configured: bool
    index_exists: bool
