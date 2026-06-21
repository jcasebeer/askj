"""
Ask J — web backend.

Serves the "Ask J" chat UI and a streaming RAG endpoint that reuses the
retrieval + Claude pipeline from bible_rag.py.

Run:
  export ANTHROPIC_API_KEY=sk-ant-...
  pip install -r requirements.txt
  uvicorn server:app --reload
  # open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import bible_rag

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Ask J")

# Loaded once at startup (download + embedding index), then reused per request.
_corpus: list[dict] = []
_embeddings = None
_model = None


@app.on_event("startup")
def _load_engine() -> None:
    global _corpus, _embeddings, _model
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY in your environment before starting.")
    _embeddings, _model, _corpus = bible_rag.load_or_build_index()


class AskRequest(BaseModel):
    question: str
    k: int = bible_rag.DEFAULT_K


@app.post("/api/ask")
def ask(req: AskRequest) -> StreamingResponse:
    passages = bible_rag.retrieve(req.question, _corpus, _embeddings, _model, req.k)

    def generate():
        # NDJSON stream: one JSON object per line.
        sources = [
            {"source": p["source"], "ref": p["ref"], "text": p["text"], "score": p["score"]}
            for p in passages
        ]
        yield json.dumps({"type": "sources", "items": sources}) + "\n"
        try:
            for text in bible_rag.stream_answer(req.question, passages):
                yield json.dumps({"type": "token", "text": text}) + "\n"
        except Exception as e:  # surface API/streaming errors to the client
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
