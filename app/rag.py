import json
from typing import Protocol

from app.models import AnswerResponse, Citation
from app.tools import ToolRegistry
from app.vector_index import LocalVectorIndex, SearchResult


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class ChatClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        ...


class RAGService:
    def __init__(
        self,
        index: LocalVectorIndex,
        embedder: Embedder,
        chat_client: ChatClient,
        tools: ToolRegistry,
        top_k: int = 5,
    ):
        self.index = index
        self.embedder = embedder
        self.chat_client = chat_client
        self.tools = tools
        self.top_k = top_k

    def answer(self, question: str) -> AnswerResponse:
        if not self.index.records:
            self.index.load()

        query_embedding = self.embedder.embed_texts([question])[0]
        results = self.index.search(query_embedding, top_k=self.top_k)
        messages = self._build_messages(question, results)

        first_response = self.chat_client.complete(messages, tools=self.tools.schemas())
        tool_results = []
        if first_response.get("tool_calls"):
            messages.append(
                {
                    "role": "assistant",
                    "content": first_response.get("content", ""),
                    "tool_calls": first_response["tool_calls"],
                }
            )
            for tool_call in first_response["tool_calls"]:
                result = self.tools.call(tool_call["name"], tool_call["arguments"])
                tool_results.append({"name": tool_call["name"], "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["name"],
                        "content": json.dumps(result),
                    }
                )
            final_response = self.chat_client.complete(messages, tools=self.tools.schemas())
            answer = final_response.get("content", "")
        else:
            answer = first_response.get("content", "")

        return AnswerResponse(
            question=question,
            answer=answer,
            citations=[
                Citation(
                    source=result.chunk.source,
                    page=result.chunk.page,
                    score=result.score,
                    text=result.chunk.text,
                )
                for result in results
            ],
            tool_results=tool_results,
        )

    def _build_messages(self, question: str, results: list[SearchResult]) -> list[dict]:
        context = "\n\n".join(
            (
                f"[{index}] Source: {result.chunk.source}, page {result.chunk.page}\n"
                f"{result.chunk.text}"
            )
            for index, result in enumerate(results, start=1)
        )
        system_prompt = (
            "You are Ally, a careful health assistant for question answering over provided "
            "health guidance. Answer in plain language, normally in 2 to 3 short sentences, "
            "unless the user asks for more detail. Use only the retrieved context and tool "
            "results. If evidence is insufficient, say so. Cite sources briefly by filename "
            "and page. Do not give emergency medical instructions beyond advising the user to "
            "contact a qualified clinician or local emergency services for urgent symptoms."
        )
        user_prompt = f"Question: {question}\n\nRetrieved context:\n{context}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
