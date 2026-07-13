# Ally

Ally is a medication and adherence system device that uses a multimodal sensor to predict patient activity for confirmation of medication use, plus NLP for communication.

## Healthwise RAG API

This repo now includes a FastAPI question-answering service over the health guidance PDFs in:

```text
/Users/sam/Documents/Ellipsis-Care/data
```

The service uses:

- Azure OpenAI for embeddings and chat responses.
- Azure Document Intelligence for PDF extraction when configured, with local PDF extraction as a fallback.
- A local persisted vector index at `.ally_index/index.json`.
- Azure Speech for audio transcription and text-to-speech.
- A function-calling tool registry ready for database-backed medication questions such as “Did I use my drug yesterday?”

## Setup

Use the existing virtual environment:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` and fill in your Azure values. `.env` is gitignored.

## Run

```bash
./.venv/bin/python -m uvicorn main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Build or rebuild the local index:

```bash
curl -X POST http://127.0.0.1:8000/ingest
```

Ask a text question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are recommended interventions for chronic disease prevention?"}'
```

Ask with audio:

```bash
curl -X POST http://127.0.0.1:8000/voice/ask \
  -F "audio=@question.wav"
```

The voice endpoint returns the transcript, answer, citations, tool results, and synthesized response audio as base64 WAV data.

## Function Calling

`app/tools.py` defines the tool registry. The first tool is:

```text
get_medication_adherence(date)
```

It currently returns a transparent “database not configured” result unless `MEDICATION_DATABASE_URL` is set. Replace the stub handler with a real database adapter when the medication/device event schema is ready.

## Tests

```bash
./.venv/bin/python -m pytest -v
./.venv/bin/python -m compileall app main.py
```
