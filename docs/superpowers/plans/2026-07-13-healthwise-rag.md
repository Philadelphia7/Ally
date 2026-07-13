# Healthwise RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI healthwise RAG service with local persisted retrieval, Azure OpenAI answering, Azure Speech voice flow, and future-ready function calling.

**Architecture:** The app is split into focused modules under `app/`. A local JSON vector index stores chunk metadata and Azure OpenAI embeddings. The RAG service retrieves chunks, optionally runs registered tools, and calls an Azure chat model with grounded context.

**Tech Stack:** Python 3.13, FastAPI, Pydantic Settings, Azure OpenAI via `openai`, Azure Speech SDK, Azure Document Intelligence SDK, PyPDF, pytest.

## Global Constraints

- Do not commit `.env`, `.venv`, generated indexes, audio files, caches, or provider outputs.
- Tests must use fake clients and must not call Azure services.
- The local vector index must persist under `.ally_index/` by default.
- The medication adherence tool must be present but transparent that database access is not configured yet.
- Commit changes in small checkpoints.

---

### Task 1: Project Scaffolding And Configuration

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings`, `get_settings()`

- [ ] Write failing config tests for default paths and env parsing.
- [ ] Run `./.venv/bin/python -m pytest tests/test_config.py -v` and verify it fails because modules are missing.
- [ ] Implement config, package scaffolding, requirements, and `.gitignore`.
- [ ] Run `./.venv/bin/python -m pytest tests/test_config.py -v` and verify it passes.
- [ ] Commit as `chore: scaffold healthwise rag app`.

### Task 2: Chunking, Local Index, And Tools

**Files:**
- Create: `app/chunking.py`
- Create: `app/vector_index.py`
- Create: `app/tools.py`
- Create: `tests/test_chunking.py`
- Create: `tests/test_vector_index.py`
- Create: `tests/test_tools.py`

**Interfaces:**
- Produces: `chunk_text`, `LocalVectorIndex`, `ToolRegistry`, `build_default_registry()`

- [ ] Write failing tests for chunk overlap, vector save/load/search, and medication tool unavailable response.
- [ ] Run targeted pytest and verify failures.
- [ ] Implement chunking, JSON vector index, cosine search, and tool registry.
- [ ] Run targeted pytest and verify passes.
- [ ] Commit as `feat: add local retrieval primitives`.

### Task 3: Azure Adapters And Document Ingestion

**Files:**
- Create: `app/azure_clients.py`
- Create: `app/document_loaders.py`
- Create: `app/ingestion.py`
- Create: `tests/test_ingestion.py`

**Interfaces:**
- Produces: `AzureOpenAIClient`, `AzureSpeechClient`, `DocumentLoader`, `IngestionService`

- [ ] Write failing ingestion tests using fake embedder and local sample text.
- [ ] Run targeted pytest and verify failures.
- [ ] Implement Azure adapters, PDF extraction fallback, Document Intelligence extraction, and ingestion service.
- [ ] Run targeted pytest and verify passes.
- [ ] Commit as `feat: add document ingestion`.

### Task 4: RAG Orchestration And FastAPI Routes

**Files:**
- Create: `app/models.py`
- Create: `app/rag.py`
- Create: `app/api.py`
- Modify: `main.py`
- Create: `tests/test_rag.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Produces: `RAGService`, `create_app()`, FastAPI `app`

- [ ] Write failing tests for grounded answers, missing index errors, tool-call routing, and route contracts.
- [ ] Run targeted pytest and verify failures.
- [ ] Implement request/response models, RAG service, routes, and entrypoint.
- [ ] Run targeted pytest and verify passes.
- [ ] Commit as `feat: expose rag fastapi service`.

### Task 5: Documentation And Verification

**Files:**
- Modify: `README.md`
- Create: `.env.example`

**Interfaces:**
- Consumes: all prior app modules and routes.

- [ ] Document setup, ingestion, text QA, voice QA, voice selection, and function-calling extension point.
- [ ] Run `./.venv/bin/python -m pytest -v`.
- [ ] Run `./.venv/bin/python -m compileall app main.py`.
- [ ] Commit as `docs: document healthwise rag service`.

