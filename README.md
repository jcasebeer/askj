# Ask J

Ask natural-language questions about the Bible and get answers grounded in
source text, with citations. Retrieval-augmented generation (RAG): local
embeddings find the most relevant passages, then Claude answers using only those
passages. Comes as a **web chat app** ("Ask J") and a **CLI**.

## How it works

1. **Corpus** — two source types, each tagged so citations stay honest:
   - **Scripture:** every verse of the public-domain King James Version (KJV).
   - **Reference:** ~11,800 entries from four public-domain Bible dictionaries
     (Easton, Smith, Hitchcock, Torrey), pulled from the
     [`JWBickel/BibleDictionaries`](https://huggingface.co/datasets/JWBickel/BibleDictionaries)
     dataset. These handle synthesis/encyclopedic questions (e.g. "who were
     Jesus's brothers?") that verse-level chunks answer poorly.
2. **Retrieval** — embeds the whole corpus (~48k chunks) once with a local
   `sentence-transformers` model (`all-MiniLM-L6-v2`), caches the vectors, and
   does cosine-similarity search for your question. No embedding API needed.
3. **Generation** — passes the top matching passages to **Claude Opus 4.8**,
   which answers *only* from them and cites each source — `(John 3:16)` for
   scripture, `(Smith's Bible Dictionary: James)` for reference entries.

Scripture is treated as the primary authority; the dictionaries (19th-century
Protestant reference works) are used for definitions, history, and context, and
flagged as such. Because the model is constrained to retrieved passages, it won't
invent content and will say when the passages don't cover your question.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Web app ("Ask J")

```bash
uvicorn server:app --reload
# open http://127.0.0.1:8000
```

A single-page chat UI. Each question streams the answer token-by-token and shows
a collapsible list of the retrieved passages, each tagged `verse` or `dict`. The
backend (`server.py`) loads the index once at startup and reuses the same
pipeline as the CLI.

## CLI

```bash
# Interactive
python bible_rag.py

# One-shot
python bible_rag.py "Who were Jesus's brothers?"

# Show which passages were retrieved, and pull more of them
python bible_rag.py --show-sources --k 12 "What are the fruits of the Spirit?"

# Rebuild the embedding cache (e.g. after changing the corpus or model)
python bible_rag.py --reindex
```

The first run downloads the text + dictionaries and builds the embedding index
(~4 minutes on CPU, one time). Subsequent runs load from `.cache/` and start
instantly.

## Notes

- **Sources:** KJV (public domain) + four public-domain Bible dictionaries.
  The corpus is built in `bible_rag.py` (`build_corpus`); bump `CORPUS_VERSION`
  after changing it so the cache rebuilds.
- **Cost:** Embeddings are computed locally and free. Only the answer step calls
  the Claude API.
- **Quality knobs:** raise `--k` to give the model more context; swap
  `EMBED_MODEL` for a larger model for better retrieval at some speed cost.
- **Known limitation:** KJV's archaic vocabulary can dilute retrieval for modern
  phrasings (e.g. "worry" → the KJV says "take no thought" / "be not afraid"), so
  some lexical matches are weaker than the semantic intent.
