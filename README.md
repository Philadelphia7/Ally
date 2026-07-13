# Ally

Ally is a medication and adherence system device that uses a multimodal sensor to predict patient activity for confirmation of medication use, plus NLP for simple patient communication.

## Healthwise RAG API

This repo includes a FastAPI service for concise health question answering over the Healthwise PDF corpus in:

```text
/Users/sam/Documents/Ellipsis-Care/data
```

The API is designed for regular, simple use such as elder-care support. RAG answers are prompted to use plain language and stay around 2 to 3 short sentences unless the user asks for more detail. The `answer` text avoids source names, page references, markdown, bullet points, and hard-to-say abbreviations so it works well for speech playback.

## What It Does

- Builds a local persisted vector index at `.ally_index/index.json`.
- Extracts PDF text with Azure Document Intelligence when configured, with local PDF extraction as fallback.
- Embeds document chunks with Azure OpenAI in safe batches.
- Answers text questions with retrieved context, separate citation metadata, and function-call results.
- Supports a full voice flow: audio question -> transcription -> RAG answer -> synthesized speech.
- Supports standalone speech-to-text and text-to-speech endpoints.
- Includes a function-calling tool registry for future database-backed medication questions such as “Did I use my drug yesterday?”

## Setup

Use the existing virtual environment:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` and fill in your Azure values. `.env` is gitignored.

Important settings:

```text
AZURE_OPENAI_API_KEY
AZURE_OPENAI_BASE_URL
AZURE_OPENAI_DEPLOYMENT_NAME
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
AZURE_OPENAI_API_VERSION
DOCUMENT_INTELLIGENCE_ENDPOINT
DOCUMENT_INTELLIGENCE_SUBSCRIPTION_KEY
SPEECH_KEY
SPEECH_REGION
SPEECH_VOICE_NAME
```

Optional settings:

```text
CHAT_TEMPERATURE
EMBEDDING_BATCH_SIZE
EMBEDDING_BATCH_MAX_CHARACTERS
MEDICATION_DATABASE_URL
```

Chat completions omit `temperature` by default because some Azure OpenAI models only support the default value. Set `CHAT_TEMPERATURE` only when your deployed model supports it.

## Run

Start the API:

```bash
./.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8081 --reload
```

Open the interactive docs:

```text
http://127.0.0.1:8081/docs
```

OpenAPI JSON:

```text
http://127.0.0.1:8081/openapi.json
```

## Endpoint Guide

### `GET /health`

Checks whether the API can see the required configuration and local index.

```bash
curl http://127.0.0.1:8081/health
```

Example response:

```json
{
  "status": "ok",
  "azure_openai_configured": true,
  "document_intelligence_configured": true,
  "speech_configured": true,
  "index_exists": true
}
```

### `POST /ingest`

Builds or rebuilds `.ally_index/index.json` from the configured PDF folder.

```bash
curl -X POST http://127.0.0.1:8081/ingest
```

Example response:

```json
{
  "document_count": 5,
  "page_count": 627,
  "chunk_count": 1663,
  "index_path": ".ally_index/index.json"
}
```

Embedding calls are batched during ingestion to stay below Azure OpenAI request limits. Tune `EMBEDDING_BATCH_SIZE` and `EMBEDDING_BATCH_MAX_CHARACTERS` in `.env` if your embedding deployment needs smaller requests.

### `POST /ask`

Answers a text question from the indexed health guidance.

```bash
curl -X POST http://127.0.0.1:8081/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are recommended interventions for chronic disease prevention?"}'
```

Response fields:

- `question`: the input question.
- `answer`: concise plain-language answer for display or speech. It should not include source names or page references.
- `citations`: retrieved source chunks with filename, page, score, and text. Use this for debugging, audit trails, or an optional “sources” view.
- `tool_results`: any function-call results, such as medication adherence checks.

### `POST /voice/ask`

Runs the full voice assistant pipeline.

```bash
curl -X POST http://127.0.0.1:8081/voice/ask \
  -F "audio=@question.wav" \
  -F "audio_format=wav"
```

Flow:

```text
audio upload -> Azure Speech transcription -> RAG answer -> Azure Speech synthesis
```

Set `audio_format` to `wav` or `mp3`. It defaults to `wav`.

Response fields include `transcript`, `answer`, `citations`, `tool_results`, `audio_base64`, `audio_content_type`, and `audio_format`.

### `POST /speech/transcribe`

Converts an audio file to text only. Use this when you need speech-to-text without asking the RAG system.

```bash
curl -X POST http://127.0.0.1:8081/speech/transcribe \
  -F "audio=@question.wav"
```

Example response:

```json
{
  "transcript": "What is diabetes?"
}
```

### `POST /speech/synthesize`

Converts text to speech only. Use this when you already have text and only need audio.

```bash
curl -X POST http://127.0.0.1:8081/speech/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"Take your medicine after food.","audio_format":"wav"}'
```

Example response:

```json
{
  "audio_base64": "<base64 audio>",
  "audio_content_type": "audio/wav",
  "audio_format": "wav"
}
```

Set `audio_format` to `wav` or `mp3`. WAV uses RIFF 24 kHz, 16-bit, mono PCM. MP3 uses 24 kHz, 48 kilobit mono MPEG audio. Decode `audio_base64` on the client to play or save the audio file.

### `GET /tools`

Lists available function-calling tools.

```bash
curl http://127.0.0.1:8081/tools
```

The first tool is:

```text
get_medication_adherence(date)
```

It currently returns a transparent “database not configured” result unless `MEDICATION_DATABASE_URL` is set. Replace the stub handler in `app/tools.py` with a real database adapter when the medication/device event schema is ready.

## Audio Notes

- Input audio is passed to Azure Speech from a temporary file.
- Output audio from `/voice/ask` and `/speech/synthesize` is JSON-safe base64.
- Supported output formats are `wav` and `mp3`.
- `wav` returns content type `audio/wav` and uses RIFF 24 kHz, 16-bit, mono PCM.
- `mp3` returns content type `audio/mpeg` and uses 24 kHz, 48 kilobit mono MPEG audio.
- Voice is controlled with `SPEECH_VOICE_NAME`, defaulting to `en-NG-EzinneNeural`.

## Project Structure

```text
app/api.py              FastAPI app and route definitions
app/azure_clients.py    Azure OpenAI and Azure Speech wrappers
app/chunking.py         Text chunk model and chunking logic
app/config.py           Environment-based settings
app/document_loaders.py PDF extraction with Document Intelligence or PyPDF
app/ingestion.py        Ingestion pipeline
app/models.py           Request and response schemas
app/rag.py              Retrieval, function calling, and answer orchestration
app/tools.py            Tool registry and medication adherence stub
app/vector_index.py     Local JSON vector index
```

## Tests

```bash
./.venv/bin/python -m pytest -v
./.venv/bin/python -m compileall app main.py
```
