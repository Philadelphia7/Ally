from dataclasses import dataclass
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from pypdf import PdfReader

from app.config import Settings


@dataclass(frozen=True)
class LoadedPage:
    source: str
    page: int
    text: str


class DocumentLoader:
    def __init__(self, settings: Settings):
        self.settings = settings

    def load_directory(self, directory: Path) -> list[LoadedPage]:
        pages: list[LoadedPage] = []
        for pdf_path in sorted(directory.glob("*.pdf")):
            pages.extend(self.load_pdf(pdf_path))
        return pages

    def load_pdf(self, path: Path) -> list[LoadedPage]:
        if self.settings.document_intelligence_configured:
            return self._load_with_document_intelligence(path)
        return self._load_with_pypdf(path)

    def _load_with_pypdf(self, path: Path) -> list[LoadedPage]:
        reader = PdfReader(str(path))
        pages: list[LoadedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(LoadedPage(source=path.name, page=index, text=text))
        return pages

    def _load_with_document_intelligence(self, path: Path) -> list[LoadedPage]:
        client = DocumentIntelligenceClient(
            endpoint=self.settings.document_intelligence_endpoint,
            credential=AzureKeyCredential(self.settings.document_intelligence_subscription_key),
        )
        with path.open("rb") as document:
            poller = client.begin_analyze_document(
                "prebuilt-read",
                body=document,
                content_type="application/pdf",
            )
            result = poller.result()

        pages: list[LoadedPage] = []
        for page in result.pages or []:
            lines = [line.content for line in page.lines or [] if line.content]
            text = "\n".join(lines)
            if text.strip():
                pages.append(
                    LoadedPage(
                        source=path.name,
                        page=page.page_number,
                        text=text,
                    )
                )
        return pages

