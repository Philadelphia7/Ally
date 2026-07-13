from types import SimpleNamespace

from app.azure_clients import (
    AzureOpenAIClient,
    build_speech_config,
    iter_embedding_batches,
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


def test_build_speech_config_uses_wav_output_format():
    settings = SimpleNamespace(
        speech_key="key",
        speech_region="westus",
        speech_voice_name="en-NG-EzinneNeural",
    )

    speech_config = build_speech_config(settings)

    assert speech_config.speech_synthesis_output_format_string == "riff-24khz-16bit-mono-pcm"
