# Healthwise RAG FastAPI Design

## Goal

Build a FastAPI service that answers health questions from the PDF corpus in `/Users/sam/Documents/Ellipsis-Care/data`, supports audio question answering with Azure Speech, and is structured for future database-backed function calls such as medication adherence checks.

## Architecture

The service will use a local persisted vector index rather than Azure AI Search. Ingestion extracts text from PDFs with Azure Document Intelligence when credentials are available, with a local PDF extraction fallback for development and tests. Text is chunked, embedded with Azure OpenAI, and stored in `.ally_index/index.json`.

Question answering will retrieve relevant chunks from the local index, call Azure OpenAI chat with grounded context and citations, and optionally execute registered tools. The first tool boundary will include a medication-adherence function that returns an explicit not-configured response until a real database adapter is connected.

Voice question answering will accept uploaded audio, transcribe it with Azure Speech, route the transcript through the same RAG/tool orchestration, then synthesize the answer with Azure Speech. Voice names are configurable so Nigerian English voices can be selected when available, with safe defaults in configuration.

## Components

- `app/config.py`: environment-driven settings and path defaults.
- `app/models.py`: request/response schemas.
- `app/document_loaders.py`: PDF extraction via Document Intelligence or local fallback.
- `app/chunking.py`: deterministic chunk creation with document metadata.
- `app/vector_index.py`: local JSON vector index with cosine retrieval.
- `app/azure_clients.py`: Azure OpenAI and Speech integration.
- `app/tools.py`: function registry and medication-adherence stub.
- `app/rag.py`: retrieval, tool routing, and answer generation.
- `app/api.py`: FastAPI routes.
- `main.py`: application entrypoint for uvicorn.

## API

- `GET /health`: verifies service configuration state without exposing secrets.
- `POST /ingest`: builds or rebuilds the local vector index from the configured data directory.
- `POST /ask`: answers a text question with citations and optional tool calls.
- `POST /voice/ask`: accepts audio upload and returns transcript, answer, citations, and synthesized audio metadata.
- `GET /tools`: lists available function tools.

## Error Handling

The API returns clear 4xx errors for missing index or invalid input, and 5xx errors for provider failures. Answers must say when the corpus does not contain enough evidence. The medication tool must not pretend a database exists; until configured, it returns a transparent unavailable result.

## Testing

Unit tests cover chunking, vector retrieval, tool registry behavior, and RAG orchestration with fake clients. API tests cover route contracts using dependency overrides where needed. Provider-heavy paths are isolated behind adapters so tests do not call Azure services.

