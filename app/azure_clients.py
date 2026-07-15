import json
import mimetypes
from pathlib import Path
from typing import Any, Literal

import azure.cognitiveservices.speech as speechsdk
import requests
from openai import AzureOpenAI

from app.config import Settings

AudioFormat = Literal["wav", "mp3"]

SUPPORTED_SPEECH_INPUT_FORMATS = "WAV, MP3, OGG Opus, or WebM Opus"


class AzureOpenAIClient:
    def __init__(self, settings: Settings):
        if not settings.azure_openai_configured:
            raise ValueError("Azure OpenAI is not configured")
        self.settings = settings
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_base_url,
            api_version=settings.azure_openai_api_version,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for batch in iter_embedding_batches(
            texts,
            max_batch_items=self.settings.embedding_batch_size,
            max_batch_characters=self.settings.embedding_batch_max_characters,
        ):
            response = self.client.embeddings.create(
                model=self.settings.azure_openai_embedding_deployment_name,
                input=batch,
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.settings.azure_openai_deployment_name,
            "messages": messages,
        }
        if self.settings.chat_temperature is not None:
            kwargs["temperature"] = self.settings.chat_temperature
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        return {
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": json.loads(call.function.arguments or "{}"),
                }
                for call in (message.tool_calls or [])
            ],
        }


class AzureSpeechClient:
    def __init__(self, settings: Settings):
        if not settings.speech_configured:
            raise ValueError("Azure Speech is not configured")
        self.settings = settings
        self.http_client = requests

    def transcribe_file(self, audio_path: Path) -> str:
        content_type = mimetypes.guess_type(audio_path.name)[0]
        return self.transcribe_audio(
            audio_path.read_bytes(),
            content_type=content_type,
            filename=audio_path.name,
        )

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        content_type: str | None = None,
        filename: str | None = None,
    ) -> str:
        if not audio_bytes:
            raise SpeechTranscriptionError("Uploaded audio is empty.")

        recognition_content_type = speech_recognition_content_type(content_type, filename)
        response = self.http_client.post(
            _speech_to_text_url(self.settings.speech_region),
            params={"language": self.settings.speech_recognition_language},
            headers={
                "Ocp-Apim-Subscription-Key": self.settings.speech_key,
                "Content-Type": recognition_content_type,
                "Accept": "application/json",
            },
            data=audio_bytes,
            timeout=60,
        )
        if response.status_code >= 400:
            raise SpeechTranscriptionError(
                f"Azure Speech transcription failed with status {response.status_code}: "
                f"{response.text[:500]}"
            )

        payload = response.json()
        status = payload.get("RecognitionStatus")
        if status != "Success":
            raise SpeechTranscriptionError(
                f"Speech was not recognized clearly. Recognition status: {status or 'unknown'}."
            )
        transcript = payload.get("DisplayText", "").strip()
        if not transcript:
            raise SpeechTranscriptionError(
                "Speech was recognized, but no transcript was returned. "
                "Please try a clearer recording in "
                f"{SUPPORTED_SPEECH_INPUT_FORMATS}; M4A/MP4 recordings are not supported."
            )
        return transcript

    def synthesize(self, text: str, audio_format: AudioFormat = "wav") -> bytes:
        speech_config = build_speech_config(self.settings, audio_format=audio_format)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None,
        )
        result = synthesizer.speak_text_async(text).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise RuntimeError(f"Speech synthesis failed: {result.reason}")
        return bytes(result.audio_data)


def build_speech_config(
    settings: Settings,
    audio_format: AudioFormat = "wav",
) -> speechsdk.SpeechConfig:
    speech_config = speechsdk.SpeechConfig(
        subscription=settings.speech_key,
        region=settings.speech_region,
    )
    speech_config.speech_synthesis_voice_name = settings.speech_voice_name
    speech_config.set_speech_synthesis_output_format(_speech_output_format(audio_format))
    return speech_config


class SpeechTranscriptionError(RuntimeError):
    pass


def speech_recognition_content_type(
    content_type: str | None,
    filename: str | None = None,
) -> str:
    normalized = (content_type or "").split(";")[0].strip().lower()
    suffix = Path(filename or "").suffix.lower()

    if suffix in {".m4a", ".mp4"} or normalized in {"audio/mp4", "audio/x-m4a", "video/mp4"}:
        raise SpeechTranscriptionError(
            "M4A/MP4 audio is not supported by this speech endpoint. "
            f"Please upload {SUPPORTED_SPEECH_INPUT_FORMATS}."
        )
    if suffix in {".opus", ".ogg"} or normalized in {"audio/ogg", "audio/opus"}:
        return "audio/ogg; codecs=opus"
    if suffix == ".webm" or normalized == "audio/webm":
        return "audio/webm; codecs=opus"
    if suffix == ".mp3" or normalized in {"audio/mpeg", "audio/mp3"}:
        return "audio/mpeg"
    if suffix == ".wav" or normalized in {"audio/wav", "audio/x-wav", "audio/wave"}:
        return "audio/wav"
    if normalized:
        return normalized
    return "audio/wav"


def _speech_to_text_url(region: str) -> str:
    return (
        f"https://{region}.stt.speech.microsoft.com/"
        "speech/recognition/conversation/cognitiveservices/v1"
    )


def content_type_for_audio_format(audio_format: AudioFormat) -> str:
    if audio_format == "wav":
        return "audio/wav"
    if audio_format == "mp3":
        return "audio/mpeg"
    raise ValueError(f"Unsupported audio format: {audio_format}")


def _speech_output_format(audio_format: AudioFormat):
    if audio_format == "wav":
        return speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    if audio_format == "mp3":
        return speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3
    raise ValueError(f"Unsupported audio format: {audio_format}")


def iter_embedding_batches(
    texts: list[str],
    max_batch_items: int,
    max_batch_characters: int,
):
    if max_batch_items <= 0:
        raise ValueError("max_batch_items must be greater than zero")
    if max_batch_characters <= 0:
        raise ValueError("max_batch_characters must be greater than zero")

    batch: list[str] = []
    batch_characters = 0
    for text in texts:
        text_characters = len(text)
        would_exceed_items = len(batch) >= max_batch_items
        would_exceed_characters = (
            batch_characters + text_characters > max_batch_characters
        )
        if batch and (would_exceed_items or would_exceed_characters):
            yield batch
            batch = []
            batch_characters = 0

        batch.append(text)
        batch_characters += text_characters

    if batch:
        yield batch
