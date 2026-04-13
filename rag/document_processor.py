"""
document_processor.py
─────────────────────
Streams PDF pages in configurable batches so arbitrarily large files (500+
pages, 200 MB) are processed without blowing up RAM.

Strategy
────────
1. Open the PDF with pypdf directly — no full-load via LangChain loader.
2. Yield page_batch_size pages at a time.
3. Split each mini-batch into chunks immediately.
4. Hand chunks to the vector store in the same batch — old batches are GC'd.

Peak memory ≈ proportional to batch_size, NOT total PDF size.
"""

import os
from typing import Generator, List, Tuple

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        page_batch_size: int = 30,       # pages processed per round-trip
    ):
        self.chunk_size      = chunk_size
        self.chunk_overlap   = chunk_overlap
        self.page_batch_size = page_batch_size

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def count_pages(self, pdf_path: str) -> int:
        """Return total page count without loading any text."""
        import pypdf
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            return len(reader.pages)

    def stream_chunks(
        self, pdf_path: str
    ) -> Generator[Tuple[List[Document], int, int], None, None]:
        """
        Generator — yields (chunk_list, pages_done, total_pages) per batch.

        Callers iterate this to ingest incrementally and update progress.
        Memory at any point ≈ page_batch_size pages worth of text.
        """
        import pypdf

        filename = os.path.basename(pdf_path)

        with open(pdf_path, "rb") as f:
            reader      = pypdf.PdfReader(f)
            total_pages = len(reader.pages)
            batch_docs  = []

            for page_idx, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                if text.strip():
                    batch_docs.append(Document(
                        page_content=text,
                        metadata={
                            "page":        page_idx,
                            "source_file": filename,
                            "source":      pdf_path,
                        },
                    ))

                # Flush every page_batch_size pages (or at EOF)
                is_last = (page_idx == total_pages - 1)
                if len(batch_docs) >= self.page_batch_size or (is_last and batch_docs):
                    chunks = self.splitter.split_documents(batch_docs)
                    yield chunks, page_idx + 1, total_pages
                    batch_docs = []          # free memory for GC

    def load_and_split(self, pdf_path: str) -> List[Document]:
        """Convenience wrapper — collects all chunks at once."""
        all_chunks = []
        for chunks, _, _ in self.stream_chunks(pdf_path):
            all_chunks.extend(chunks)
        return all_chunks

    def get_stats(self, chunks: List[Document]) -> dict:
        total_chars = sum(len(c.page_content) for c in chunks)
        pages = set(c.metadata.get("page", 0) for c in chunks)
        return {
            "total_chunks":     len(chunks),
            "total_characters": total_chars,
            "unique_pages":     len(pages),
            "avg_chunk_size":   total_chars // max(len(chunks), 1),
        }
