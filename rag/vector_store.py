"""
vector_store.py
───────────────
ChromaDB manager with batch ingestion support for large PDFs.

Embedding model
───────────────
  Name   : sentence-transformers/all-MiniLM-L6-v2
  Dims   : 384
  Device : CPU  (no GPU needed)
  Speed  : ~14 000 tokens/sec on CPU
  Source : HuggingFace Hub (auto-downloaded ~90 MB on first run)
"""

import os
from typing import List, Callable, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


CHROMA_PERSIST_DIR   = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# How many chunks to embed + upsert in a single ChromaDB call.
# Larger = fewer round-trips but more RAM.  500 is safe up to ~8 GB RAM.
EMBED_BATCH_SIZE = 500


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32   # reduce from 64 → safer for RAM
        },
    )

class VectorStoreManager:
    def __init__(self):
        self.embeddings = get_embeddings()
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _collection_name(self, key: str) -> str:
        name = os.path.splitext(key)[0]
        return "".join(c if c.isalnum() else "_" for c in name)[:63]

    def _open_or_create(self, collection: str) -> Chroma:
        return Chroma(
            collection_name=collection,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, chunks: List[Document], key: str) -> Chroma:
        """
        Simple single-call ingestion — use for small PDFs or as fallback.
        """
        collection = self._collection_name(key)
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection,
            persist_directory=CHROMA_PERSIST_DIR,
        )

    def ingest_batch(
        self,
        chunks: List[Document],
        key: str,
        vectordb: Optional[Chroma] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Chroma:
        """
        Batched ingestion — safe for any PDF size.

        Parameters
        ----------
        chunks      : list of Document objects (one batch from stream_chunks)
        key         : collection identifier (session_id + filename)
        vectordb    : existing Chroma instance to append to (None = create)
        progress_cb : optional callable(chunks_done, chunks_total)

        Returns the Chroma instance (create or existing).
        """
        collection = self._collection_name(key)

        if vectordb is None:
            vectordb = self._open_or_create(collection)

        total = len(chunks)
        for start in range(0, total, EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            vectordb.add_documents(batch)
            if progress_cb:
                progress_cb(min(start + EMBED_BATCH_SIZE, total), total)

        return vectordb

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, key: str) -> Chroma:
        collection = self._collection_name(key)
        return Chroma(
            collection_name=collection,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def similarity_search(self, vectordb: Chroma, query: str, k: int = 4) -> list:
        query = f"Represent this sentence for searching: {query}"
        return vectordb.similarity_search_with_relevance_scores(query, k=k)
    
    
    # ── Admin ─────────────────────────────────────────────────────────────────

    def list_collections(self) -> List[str]:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        return [c.name for c in client.list_collections()]

    def delete_collection(self, key: str):
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        try:
            client.delete_collection(self._collection_name(key))
        except Exception:
            pass
