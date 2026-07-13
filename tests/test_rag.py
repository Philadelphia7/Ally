from app.chunking import TextChunk
from app.rag import RAGService
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

