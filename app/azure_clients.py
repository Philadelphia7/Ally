import json
from pathlib import Path
from typing import Any

import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

from app.config import Settings


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
        response = self.client.embeddings.create(
            model=self.settings.azure_openai_embedding_deployment_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.settings.azure_openai_deployment_name,
            "messages": messages,
            "temperature": 0.1,
        }
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
        speech_config = speechsdk.SpeechConfig(
            subscription=self.settings.speech_key,
            region=self.settings.speech_region,
        )
        audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )
        result = recognizer.recognize_once_async().get()
        if result.reason != speechsdk.ResultReason.RecognizedSpeech:
            raise RuntimeError(f"Speech transcription failed: {result.reason}")
        return result.text

    def synthesize(self, text: str) -> bytes:
        speech_config = speechsdk.SpeechConfig(
            subscription=self.settings.speech_key,
            region=self.settings.speech_region,
        )
        speech_config.speech_synthesis_voice_name = self.settings.speech_voice_name
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None,
        )
        result = synthesizer.speak_text_async(text).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise RuntimeError(f"Speech synthesis failed: {result.reason}")
        return bytes(result.audio_data)

