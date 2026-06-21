"""
Ask J — RAG over the Bible, answered from retrieved scripture and reference works.

Corpus (each chunk tagged with its source):
  - Scripture: every verse of the public-domain KJV.
  - Reference: entries from four public-domain Bible dictionaries
    (Easton, Hitchcock, Smith, Torrey) — good for synthesis / encyclopedic
    questions that verse-level chunks answer poorly.

Pipeline:
  1. Build the corpus (download + cache KJV; download dictionaries via HF).
  2. Embed every chunk once with a local sentence-transformers model (cached).
  3. For a question: embed it, cosine-similarity search for the top-k chunks.
  4. Hand those to Claude, which answers ONLY from them and cites each source.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python bible_rag.py                       # interactive REPL
  python bible_rag.py "Who were Jesus's brothers?"
  python bible_rag.py --k 12 "..."          # retrieve more chunks
  python bible_rag.py --reindex             # force-rebuild the embedding cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / ".cache"
BIBLE_JSON = CACHE_DIR / "kjv.json"
EMB_FILE = CACHE_DIR / "embeddings.npy"
CORPUS_FILE = CACHE_DIR / "corpus.json"      # persisted alongside vectors (kept aligned)
META_FILE = CACHE_DIR / "index_meta.json"

# Bump when the corpus composition or chunking logic changes, to force a rebuild.
CORPUS_VERSION = "v2-dictionaries"

# Public-domain King James Version, structured as books -> chapters -> verses.
BIBLE_URL = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"

# Public-domain Bible dictionaries on Hugging Face (one config per dictionary).
DICT_DATASET = "JWBickel/BibleDictionaries"
DICTIONARIES = {
    "Easton": "Easton's Bible Dictionary",
    "Smith": "Smith's Bible Dictionary",
    "Hitchcock": "Hitchcock's Bible Names Dictionary",
    "Torrey": "Torrey's Topical Textbook",
}

CHUNK_WORDS = 130                 # split long dictionary entries to fit the embedder
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, fast, good enough for retrieval
CLAUDE_MODEL = "claude-haiku-4-5"
DEFAULT_K = 10

# Adaptive thinking is a 4.6+/Fable feature. Older-generation models (Haiku 4.5,
# Sonnet 4.5) reject `thinking: {type: "adaptive"}` with a 400, so only send it
# for models that support it.
ADAPTIVE_THINKING_MODELS = {
    "claude-fable-5", "claude-opus-4-8", "claude-opus-4-7",
    "claude-opus-4-6", "claude-sonnet-4-6",
}

SYSTEM_PROMPT = """You are a careful, respectful study assistant for questions about the Bible.

You are given a user's question and a set of retrieved passages. Each passage is \
labeled with its source:
- Scripture references (e.g. "John 3:16") are from the King James Version and are \
the primary authority for what the Bible says.
- "...Bible Dictionary" / "Topical Textbook" entries are 19th-century public-domain \
Protestant reference works. Use them for definitions, genealogy, history, and \
context, but treat them as scholarship of their era and tradition, not as \
infallible.

Answer using ONLY the retrieved passages.

Rules:
- Ground every claim in the passages and cite the source inline, e.g. \
(Matthew 13:55) or (Smith's Bible Dictionary: Brethren of the Lord).
- Prefer scripture for what the Bible says; use the dictionary entries to define, \
contextualize, or synthesize.
- If the passages do not contain enough to answer, say so plainly rather than \
inventing content or relying on outside knowledge.
- When a question touches matters on which Christian traditions differ (e.g. the \
nature of Jesus' "brothers"), present the differing views fairly without taking a side.
- Be concise and clear; quote scripture accurately and sparingly.
"""


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

def _chunk(text: str, max_words: int = CHUNK_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def load_bible() -> list[dict]:
    """Return one {"source","ref","text"} item per KJV verse."""
    CACHE_DIR.mkdir(exist_ok=True)
    if not BIBLE_JSON.exists():
        print(f"Downloading KJV text from {BIBLE_URL} ...", file=sys.stderr)
        resp = requests.get(BIBLE_URL, timeout=60)
        resp.raise_for_status()
        BIBLE_JSON.write_bytes(resp.content)  # UTF-8 with BOM; decoded below

    raw = json.loads(BIBLE_JSON.read_text(encoding="utf-8-sig"))

    verses: list[dict] = []
    for book in raw:
        name = book["name"]
        for c_idx, chapter in enumerate(book["chapters"], start=1):
            for v_idx, text in enumerate(chapter, start=1):
                verses.append({
                    "source": "scripture",
                    "ref": f"{name} {c_idx}:{v_idx}",
                    "text": text.strip(),
                })
    return verses


def load_dictionaries() -> list[dict]:
    """Return chunked dictionary entries, each tagged with its dictionary name."""
    from datasets import load_dataset

    items: list[dict] = []
    for config, full_name in DICTIONARIES.items():
        print(f"Loading {full_name} ...", file=sys.stderr)
        ds = load_dataset(DICT_DATASET, config, split="train")
        for row in ds:
            term = (row.get("term") or "").strip()
            defs = row.get("definitions") or []
            body = "\n".join(d.strip() for d in defs if d and d.strip())
            if not term or not body:
                continue
            for chunk in _chunk(body):
                items.append({
                    "source": "dictionary",
                    "ref": f"{full_name}: {term}",
                    # Prepend the term so the embedding and Claude both see the subject.
                    "text": f"{term}. {chunk}",
                })
    return items


def build_corpus() -> list[dict]:
    corpus = load_bible() + load_dictionaries()
    print(f"Corpus: {len(corpus)} chunks total.", file=sys.stderr)
    return corpus


# ---------------------------------------------------------------------------
# Embedding index
# ---------------------------------------------------------------------------

def load_or_build_index(reindex: bool = False):
    """Return (embeddings[N,D] L2-normalized, model, corpus). Cached to disk.

    The corpus is persisted next to the vectors so the two never drift out of
    alignment, and so the fast path needs no network (HF/GitHub) access.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)

    if not reindex and EMB_FILE.exists() and CORPUS_FILE.exists() and META_FILE.exists():
        meta = json.loads(META_FILE.read_text())
        if meta.get("version") == CORPUS_VERSION and meta.get("model") == EMBED_MODEL:
            return np.load(EMB_FILE), model, json.loads(CORPUS_FILE.read_text())

    corpus = build_corpus()
    print(f"Embedding {len(corpus)} chunks with {EMBED_MODEL} (one-time, a few min)...",
          file=sys.stderr)
    emb = model.encode(
        [c["text"] for c in corpus],
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,  # so dot product == cosine similarity
    ).astype(np.float32)

    np.save(EMB_FILE, emb)
    CORPUS_FILE.write_text(json.dumps(corpus))
    META_FILE.write_text(json.dumps(
        {"version": CORPUS_VERSION, "model": EMBED_MODEL, "count": len(corpus)}))
    return emb, model, corpus


def retrieve(question: str, corpus, embeddings, model, k: int) -> list[dict]:
    q = model.encode([question], normalize_embeddings=True).astype(np.float32)[0]
    scores = embeddings @ q  # cosine similarity, embeddings are normalized
    top = np.argsort(-scores)[:k]
    return [{**corpus[i], "score": float(scores[i])} for i in top]


# ---------------------------------------------------------------------------
# Answering with Claude
# ---------------------------------------------------------------------------

def _build_user_message(question: str, passages: list[dict]) -> str:
    context = "\n".join(f"[{p['ref']}] {p['text']}" for p in passages)
    return f"Retrieved passages:\n{context}\n\nQuestion: {question}"


def stream_answer(question: str, passages: list[dict]):
    """Yield answer text chunks from Claude. Reused by both the CLI and the web server."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    kwargs = dict(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(question, passages)}],
    )
    if CLAUDE_MODEL in ADAPTIVE_THINKING_MODELS:
        kwargs["thinking"] = {"type": "adaptive"}

    with client.messages.stream(**kwargs) as stream:
        yield from stream.text_stream


def answer(question: str, passages: list[dict]) -> None:
    for text in stream_answer(question, passages):
        print(text, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about the Bible (RAG).")
    parser.add_argument("question", nargs="*", help="Your question. Omit for interactive mode.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of chunks to retrieve.")
    parser.add_argument("--show-sources", action="store_true",
                        help="Print the retrieved passages before the answer.")
    parser.add_argument("--reindex", action="store_true",
                        help="Force-rebuild the embedding cache.")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Error: set ANTHROPIC_API_KEY in your environment first.")

    embeddings, model, corpus = load_or_build_index(reindex=args.reindex)

    def handle(q: str) -> None:
        passages = retrieve(q, corpus, embeddings, model, args.k)
        if args.show_sources:
            print("\n--- Retrieved passages ---", file=sys.stderr)
            for p in passages:
                print(f"  ({p['score']:.2f}) [{p['source']}] {p['ref']}", file=sys.stderr)
            print("--- Answer ---\n", file=sys.stderr)
        answer(q, passages)

    if args.question:
        handle(" ".join(args.question))
        return

    print("Ask J — ask a question about the Bible (Ctrl-D or 'quit' to exit).")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            print()
            break
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            handle(q)


if __name__ == "__main__":
    main()
