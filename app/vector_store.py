"""
Vector store layer, backed by Chroma (local, persistent, file-based).

Responsibilities:
  - chunk_text: simple splitter for long docs
  - upsert_docs: embed + store KB documents
  - retrieve: embed a query and return the top-k most relevant chunks

This is the "R" (retrieval) in the RAG pipeline used by /chat.

Embeddings: by default (USE_LOCAL_EMBEDDINGS=true, the default in .env.example)
this runs entirely locally via Chroma's bundled ONNX MiniLM-L6-v2 model --
no API key, no cost, no network calls at query time (the model itself is
downloaded once, on first use, from a public CDN). This means retrieval works
even if your CHAT_API_KEY is an OpenRouter key, which does not support an
embeddings endpoint. Set USE_LOCAL_EMBEDDINGS=false to instead call OpenAI's
embedding API via app.llm.embed_texts (requires a real, billed OpenAI key).
"""
from typing import List, Dict

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from app.config import settings

_chroma_client = chromadb.PersistentClient(
    path=settings.CHROMA_PERSIST_DIR,
    settings=ChromaSettings(anonymized_telemetry=False),
)

# Local, free, no-API-key embedding function (downloads a small ONNX model
# from a public CDN the first time it's used, then runs fully offline).
_local_embedding_fn = embedding_functions.DefaultEmbeddingFunction()

_collection = _chroma_client.get_or_create_collection(
    name=settings.CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
    embedding_function=_local_embedding_fn if settings.USE_LOCAL_EMBEDDINGS else None,
)


def _embed(texts: List[str]) -> List[List[float]]:
    if settings.USE_LOCAL_EMBEDDINGS:
        return _local_embedding_fn(texts)
    # Lazy import so app.llm (and its OpenAI client init) is only touched
    # when OpenAI-based embeddings are actually requested.
    from app.llm import embed_texts
    return embed_texts(texts)


def chunk_text(text: str, max_chars: int = 600) -> List[str]:
    """Naive fixed-size chunker with sentence-ish boundaries. Good enough for
    short FAQ articles; swap for a token-aware splitter for larger corpora."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = ""
    for sentence in text.replace("\n", " ").split(". "):
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}." if current else f"{sentence}."
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = f"{sentence}."
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return chunks


def upsert_docs(docs: List[Dict]) -> int:
    """
    docs: list of {"doc_id": str, "title": str, "text": str}
    Chunks each doc, embeds all chunks in one batch call, and upserts into Chroma.
    Returns number of chunks stored.
    """
    ids, texts, metadatas = [], [], []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            chunk_id = f"{doc['doc_id']}-{i}"
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({"doc_id": doc["doc_id"], "title": doc["title"]})

    if not texts:
        return 0

    if settings.USE_LOCAL_EMBEDDINGS:
        # Collection already has an embedding_function attached, so it will
        # embed these documents itself.
        _collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
    else:
        embeddings = _embed(texts)
        _collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(texts)


def retrieve(query: str, top_k: int = None) -> List[Dict]:
    """Embed the query and return the top-k most relevant chunks with scores."""
    top_k = top_k or settings.RETRIEVAL_TOP_K
    if _collection.count() == 0:
        return []

    if settings.USE_LOCAL_EMBEDDINGS:
        results = _collection.query(
            query_texts=[query],
            n_results=min(top_k, _collection.count()),
        )
    else:
        query_embedding = _embed([query])[0]
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, _collection.count()),
        )

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc_text, meta, distance in zip(docs, metas, distances):
        # Chroma cosine distance -> similarity score (higher = more relevant)
        similarity = 1 - distance
        chunks.append(
            {
                "doc_id": meta.get("doc_id", "unknown"),
                "title": meta.get("title", "Untitled"),
                "text": doc_text,
                "score": round(float(similarity), 4),
            }
        )
    return chunks


def collection_count() -> int:
    return _collection.count()