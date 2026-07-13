import json
from pathlib import Path
from typing import Any, Literal

import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

from app.config import Settings

AudioFormat = Literal["wav", "mp3"]


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

    def transcribe_file(self, audio_path: Path) -> str:
        speech_config = build_speech_config(self.settings)
        audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )
        result = recognizer.recognize_once_async().get()
        if result.reason != speechsdk.ResultReason.RecognizedSpeech:
            raise RuntimeError(f"Speech transcription failed: {result.reason}")
        return result.text

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
