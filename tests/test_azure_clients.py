from types import SimpleNamespace

import pytest

from app.azure_clients import (
    AzureOpenAIClient,
    AzureSpeechClient,
    SpeechTranscriptionError,
    build_speech_config,
    content_type_for_audio_format,
    iter_embedding_batches,
    speech_recognition_content_type,
)


def test_iter_embedding_batches_respects_item_and_character_limits():
    texts = ["a" * 5, "b" * 5, "c" * 9, "d" * 2]

    batches = list(
        iter_embedding_batches(
            texts,
            max_batch_items=2,
            max_batch_characters=10,
        )
    )

    assert batches == [["a" * 5, "b" * 5], ["c" * 9], ["d" * 2]]


def test_iter_embedding_batches_keeps_oversized_single_text():
    batches = list(
        iter_embedding_batches(
            ["x" * 25, "y"],
            max_batch_items=10,
            max_batch_characters=10,
        )
    )

    assert batches == [["x" * 25], ["y"]]


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="answer", tool_calls=None)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def test_complete_omits_temperature_by_default():
    completions = FakeCompletions()
    client = object.__new__(AzureOpenAIClient)
    client.settings = SimpleNamespace(
        azure_openai_deployment_name="chat-deployment",
        chat_temperature=None,
    )
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )

    response = client.complete([{"role": "user", "content": "hello"}])

    assert response["content"] == "answer"
    assert "temperature" not in completions.kwargs


def test_complete_includes_temperature_when_configured():
    completions = FakeCompletions()
    client = object.__new__(AzureOpenAIClient)
    client.settings = SimpleNamespace(
        azure_openai_deployment_name="chat-deployment",
        chat_temperature=1.0,
    )
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )

    client.complete([{"role": "user", "content": "hello"}])

    assert completions.kwargs["temperature"] == 1.0


def test_build_speech_config_uses_wav_output_format_by_default():
    settings = SimpleNamespace(
        speech_key="key",
        speech_region="westus",
        speech_voice_name="en-NG-EzinneNeural",
    )

    speech_config = build_speech_config(settings)

    assert speech_config.speech_synthesis_output_format_string == "riff-24khz-16bit-mono-pcm"


def test_build_speech_config_supports_mp3_output_format():
    settings = SimpleNamespace(
        speech_key="key",
        speech_region="westus",
        speech_voice_name="en-NG-EzinneNeural",
    )

    speech_config = build_speech_config(settings, audio_format="mp3")

    assert speech_config.speech_synthesis_output_format_string == "audio-24khz-48kbitrate-mono-mp3"
    assert content_type_for_audio_format("mp3") == "audio/mpeg"


class FakeSpeechRestResponse:
    status_code = 200
    text = '{"RecognitionStatus":"Success","DisplayText":"What is diabetes?"}'

    def json(self):
        return {"RecognitionStatus": "Success", "DisplayText": "What is diabetes?"}


class FakeHttpClient:
    def __init__(self):
        self.calls = []

    def post(self, url, params, headers, data, timeout):
        self.calls.append(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "data": data,
                "timeout": timeout,
            }
        )
        return FakeSpeechRestResponse()


def test_transcribe_audio_uses_rest_with_ogg_opus_content_type():
    http_client = FakeHttpClient()
    client = object.__new__(AzureSpeechClient)
    client.settings = SimpleNamespace(
        speech_key="key",
        speech_region="eastus",
        speech_recognition_language="en-NG",
    )
    client.http_client = http_client

    transcript = client.transcribe_audio(
        b"opus-bytes",
        content_type="audio/ogg",
        filename="question_who.opus",
    )

    assert transcript == "What is diabetes?"
    call = http_client.calls[0]
    assert call["url"] == "https://eastus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
    assert call["params"] == {"language": "en-NG"}
    assert call["headers"]["Content-Type"] == "audio/ogg; codecs=opus"
    assert call["headers"]["Ocp-Apim-Subscription-Key"] == "key"
    assert call["data"] == b"opus-bytes"


def test_speech_recognition_content_type_normalizes_opus_uploads():
    assert (
        speech_recognition_content_type("audio/ogg", "question_who.opus")
        == "audio/ogg; codecs=opus"
    )


def test_transcribe_audio_rejects_m4a_uploads_before_calling_speech_api():
    http_client = FakeHttpClient()
    client = object.__new__(AzureSpeechClient)
    client.settings = SimpleNamespace(
        speech_key="key",
        speech_region="eastus",
        speech_recognition_language="en-NG",
    )
    client.http_client = http_client

    with pytest.raises(SpeechTranscriptionError) as exc:
        client.transcribe_audio(
            b"m4a-bytes",
            content_type="audio/mp4",
            filename="test rec 1.m4a",
        )

    assert "M4A/MP4 audio is not supported" in str(exc.value)
    assert "WAV" in str(exc.value)
    assert http_client.calls == []
