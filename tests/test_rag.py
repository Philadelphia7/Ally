from app.chunking import TextChunk
from app.rag import RAGService, clean_spoken_answer
from app.tools import build_default_registry
from app.vector_index import LocalVectorIndex


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeChatClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def complete(self, messages, tools=None):
        self.messages.append(messages)
        return self.responses.pop(0)


def test_rag_answer_includes_citations_from_retrieved_context(tmp_path):
    index = LocalVectorIndex(tmp_path / "index.json")
    index.add(TextChunk("guide.pdf:1:0", "guide.pdf", 1, "Use nutrition guidance."), [1.0, 0.0])
    index.save()
    chat = FakeChatClient([{"content": "Follow the nutrition guidance.", "tool_calls": []}])
    service = RAGService(
        index=index,
        embedder=FakeEmbedder(),
        chat_client=chat,
        tools=build_default_registry(None),
    )

    answer = service.answer("What should I do?")

    assert answer.answer == "Follow the nutrition guidance."
    assert answer.citations[0].source == "guide.pdf"
    assert answer.citations[0].page == 1
    assert "Use nutrition guidance." in chat.messages[0][1]["content"]


def test_rag_system_prompt_requests_concise_plain_language(tmp_path):
    index = LocalVectorIndex(tmp_path / "index.json")
    index.add(TextChunk("guide.pdf:1:0", "guide.pdf", 1, "Use nutrition guidance."), [1.0, 0.0])
    index.save()
    chat = FakeChatClient([{"content": "Follow the nutrition guidance.", "tool_calls": []}])
    service = RAGService(
        index=index,
        embedder=FakeEmbedder(),
        chat_client=chat,
        tools=build_default_registry(None),
    )

    service.answer("What should I do?")

    system_prompt = chat.messages[0][0]["content"]
    assert "2 to 3 short sentences" in system_prompt
    assert "plain language" in system_prompt
    assert "Do not give a generic answer" in system_prompt
    assert "accurate and specific" in system_prompt
    assert "Do not include source names, page numbers, citations" in system_prompt


def test_rag_cleans_citations_and_audio_unfriendly_abbreviations(tmp_path):
    index = LocalVectorIndex(tmp_path / "index.json")
    index.add(TextChunk("guide.pdf:1:0", "guide.pdf", 1, "Use nutrition guidance."), [1.0, 0.0])
    index.save()
    chat = FakeChatClient(
        [
            {
                "content": (
                    "Recommended steps include healthy diets (e.g., less trans-fat), "
                    "physical activity, and stopping tobacco use (WHO_PEN p120; WHO_DIET p17)."
                ),
                "tool_calls": [],
            }
        ]
    )
    service = RAGService(
        index=index,
        embedder=FakeEmbedder(),
        chat_client=chat,
        tools=build_default_registry(None),
    )

    answer = service.answer("What should I do?")

    assert answer.answer == (
        "Recommended steps include healthy diets, for example, less trans-fat, "
        "physical activity, and stopping tobacco use."
    )
    assert answer.citations[0].source == "guide.pdf"


def test_clean_spoken_answer_removes_markdown_source_codes_and_page_references():
    answer = clean_spoken_answer(
        "**Use primary care screening** and support self-care (WHO_PEN p120; WHO_DIET p17). "
        "This lowers risk e.g. through better diet."
    )

    assert answer == (
        "Use primary care screening and support self-care. "
        "This lowers risk for example through better diet."
    )


def test_rag_executes_tool_calls_before_final_answer(tmp_path):
    index = LocalVectorIndex(tmp_path / "index.json")
    index.add(TextChunk("guide.pdf:1:0", "guide.pdf", 1, "Medication adherence matters."), [1.0, 0.0])
    index.save()
    chat = FakeChatClient(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "get_medication_adherence",
                        "arguments": {"date": "yesterday"},
                    }
                ],
            },
            {"content": "The database is not configured yet.", "tool_calls": []},
        ]
    )
    service = RAGService(
        index=index,
        embedder=FakeEmbedder(),
        chat_client=chat,
        tools=build_default_registry(None),
    )

    answer = service.answer("Did I use my drug yesterday?")

    assert answer.answer == "The database is not configured yet."
    assert answer.tool_results[0]["name"] == "get_medication_adherence"
    assert answer.tool_results[0]["result"]["status"] == "unavailable"


def test_rag_requires_existing_index(tmp_path):
    service = RAGService(
        index=LocalVectorIndex(tmp_path / "missing.json"),
        embedder=FakeEmbedder(),
        chat_client=FakeChatClient([]),
        tools=build_default_registry(None),
    )

    try:
        service.answer("What is diabetes?")
    except FileNotFoundError as exc:
        assert "Vector index" in str(exc)
    else:
        raise AssertionError("Expected missing index error")
