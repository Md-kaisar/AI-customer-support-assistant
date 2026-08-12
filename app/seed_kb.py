"""
Run this once (and any time the knowledge base content changes) to embed
the FAQ docs and load them into the local Chroma vector store.

Usage:
    python -m app.seed_kb
"""
from app.knowledge_base.faq_docs import FAQ_DOCS
from app.vector_store import upsert_docs, collection_count


def main():
    print(f"Existing chunks in vector store before seeding: {collection_count()}")
    n = upsert_docs(FAQ_DOCS)
    print(f"Embedded and upserted {n} chunks from {len(FAQ_DOCS)} FAQ docs.")
    print(f"Total chunks in vector store now: {collection_count()}")


if __name__ == "__main__":
    main()
