"""Document Manager — parse, store, and query uploaded documents."""

from __future__ import annotations

import asyncio
import csv
import io
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any

from src import ScriptTool
from pydantic import BaseModel, Field

_CHUNK_WORDS = 4000

_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".log", ".csv", ".docx", ".doc"}

_MIME_MAP = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".log": "text/plain",
    ".csv": "text/csv",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
}


def _chunk_text(text: str, max_words: int = _CHUNK_WORDS) -> list[str]:
    """Split text into chunks of approximately max_words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks or [""]


def _extract_pdf(file_path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _extract_docx(file_path: str) -> str:
    from docx import Document

    doc = Document(file_path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_text_file(file_path: str) -> str:
    with open(file_path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_csv(file_path: str) -> str:
    with open(file_path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return ""
    # Format as text table
    buf = io.StringIO()
    for row in rows:
        buf.write(" | ".join(row) + "\n")
    return buf.getvalue()


def _extract_text(file_path: str, ext: str) -> str:
    if ext == ".pdf":
        return _extract_pdf(file_path)
    if ext == ".docx":
        return _extract_docx(file_path)
    if ext == ".csv":
        return _extract_csv(file_path)
    # .txt, .md, .log, .doc (plain text fallback for .doc)
    return _extract_text_file(file_path)


class Input(BaseModel):
    command: str = Field(description="parse|get|list|search|chunk-embed|check-chunks|dump")
    file_path: str = Field(default="", description="Path to document file (for parse)")
    doc_id: str = Field(default="", description="Document ID (for get/chunk-embed/check-chunks/dump)")
    query: str = Field(default="", description="Search query (for search)")
    limit: int = Field(default=20, description="Max results")
    force: bool = Field(default=False, description="Force re-embedding (for chunk-embed)")


class Output(BaseModel):
    success: bool = True
    item: dict[str, Any] = Field(default_factory=dict)
    items: list[dict[str, Any]] = Field(default_factory=list)
    output_path: str = ""
    chars: int = 0
    error: str = ""


class DocumentManagerTool(ScriptTool[Input, Output]):
    name = "document_manager"
    description = "Parse, store, and query uploaded documents (PDF, DOCX, TXT, CSV, MD)"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        if inp.command == "parse":
            return await self._parse(inp.file_path)
        if inp.command == "get":
            return await self._get(inp.doc_id)
        if inp.command == "list":
            return await self._list(inp.limit)
        if inp.command == "search":
            return await self._search(inp.query, inp.limit)
        if inp.command == "chunk-embed":
            return await self._chunk_embed(inp.doc_id, inp.force)
        if inp.command == "check-chunks":
            return await self._check_chunks(inp.doc_id)
        if inp.command == "dump":
            return await self._dump(inp.doc_id)
        return Output(success=False, error=f"Unknown command: {inp.command}")

    async def _dump(self, doc_id: str) -> Output:
        """Write extracted_text to data/doc_extracts/{doc_id}.txt and return the path.

        Use this when you need to read a large document's full content — tool output
        is truncated by the runtime, but the Read tool can paginate the file.
        """
        if not doc_id:
            return Output(success=False, error="--doc_id required")

        from app.db.repos.documents import DocumentsRepo

        repo = DocumentsRepo()
        doc = await repo.get(doc_id)
        if not doc:
            return Output(success=False, error=f"document {doc_id} not found")

        text = doc.get("extracted_text", "") or ""
        if not text:
            return Output(success=False, error="document has no extracted_text")

        root = Path(__file__).resolve().parents[4]
        out_dir = root / "data" / "doc_extracts"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{doc_id}.txt"
        out_file.write_text(text, encoding="utf-8")
        rel_path = str(out_file.relative_to(root))
        return Output(success=True, output_path=rel_path, chars=len(text))

    async def _parse(self, file_path: str) -> Output:
        if not file_path:
            return Output(success=False, error="--file_path required")

        # Resolve path
        path = Path(file_path)
        if not path.is_absolute():
            from app.settings import get_settings

            path = Path(get_settings().project_root) / path

        if not path.exists():
            return Output(success=False, error=f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            return Output(
                success=False,
                error=f"Unsupported file type: {ext}. Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}",
            )

        # Extract text
        try:
            text = _extract_text(str(path), ext)
        except Exception as e:
            return Output(success=False, error=f"Extraction failed: {e}")

        # Chunk
        chunks = _chunk_text(text)
        file_size = path.stat().st_size
        mime_type = _MIME_MAP.get(ext, mimetypes.guess_type(str(path))[0] or "")

        # Generate doc_id
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d')}_{abs(hash(str(path))) % 0xFFFFFFFF:08x}"

        # Compute relative path for storage
        try:
            from app.settings import get_settings

            rel_path = str(path.relative_to(get_settings().project_root))
        except (ValueError, Exception):
            rel_path = str(path)

        # Save to DB
        from app.db.repos.documents import DocumentsRepo

        repo = DocumentsRepo()
        await repo.create(
            doc_id,
            path.name,
            file_path=rel_path,
            mime_type=mime_type,
            file_size_bytes=file_size,
            extracted_text=text,
            chunk_count=len(chunks),
        )

        # Write to context cache
        try:
            from app.db.repos.context_cache import add_to_cache

            summary = f"Document uploaded: '{path.name}' ({len(text)} chars, {len(chunks)} chunks)"
            await add_to_cache(
                "document",
                summary,
                entity_type="documents",
                entity_id=doc_id,
                file_path=rel_path,
                tags=["document", ext.lstrip(".")],
                source_agent="document_manager",
            )
        except Exception:
            pass  # Cache write is best-effort

        return Output(
            success=True,
            item={
                "doc_id": doc_id,
                "filename": path.name,
                "file_path": rel_path,
                "mime_type": mime_type,
                "file_size_bytes": file_size,
                "text_length": len(text),
                "chunk_count": len(chunks),
            },
        )

    async def _get(self, doc_id: str) -> Output:
        if not doc_id:
            return Output(success=False, error="--doc_id required")

        from app.db.repos.documents import DocumentsRepo

        record = await DocumentsRepo().get(doc_id)
        if not record:
            return Output(success=False, error=f"Not found: {doc_id}")
        record["text_length"] = len(record.get("extracted_text", "") or "")
        return Output(success=True, item=record)

    async def _list(self, limit: int) -> Output:
        from app.db.repos.documents import DocumentsRepo

        items = await DocumentsRepo().list_recent(limit=limit)
        return Output(success=True, items=items)

    async def _search(self, query: str, limit: int) -> Output:
        if not query:
            return Output(success=False, error="--query required")

        from app.db.repos.documents import DocumentsRepo

        items = await DocumentsRepo().search(query, limit=limit)
        return Output(success=True, items=items)

    async def _chunk_embed(self, doc_id: str, force: bool = False) -> Output:
        if not doc_id:
            return Output(success=False, error="--doc_id required")

        from app.db.repos.documents import DocumentsRepo
        from app.db.repos.embedding_chunks import EmbeddingChunksRepo

        chunks_repo = EmbeddingChunksRepo()

        # Check if already chunked (unless force)
        if not force:
            existing = await chunks_repo.count_for_source("documents", doc_id)
            if existing > 0:
                return Output(
                    success=True,
                    item={"doc_id": doc_id, "chunk_count": existing, "already_existed": True},
                )

        # Read document text
        record = await DocumentsRepo().get(doc_id)
        if not record:
            return Output(success=False, error=f"Not found: {doc_id}")

        text = record.get("extracted_text", "")
        if not text or not text.strip():
            return Output(success=False, error=f"Document {doc_id} has no extracted text")

        # Chunk text using embeddings service
        from app.services.embeddings import chunk_text, get_embeddings_batch

        text_chunks = chunk_text(text)
        if not text_chunks:
            return Output(success=False, error=f"Document {doc_id} produced no chunks")

        # Embed all chunks (batch — runs synchronous OpenAI call in thread)
        embeddings = await asyncio.to_thread(get_embeddings_batch, text_chunks)

        # Store via EmbeddingChunksRepo
        chunk_data = [(i, t, e) for i, (t, e) in enumerate(zip(text_chunks, embeddings, strict=True))]
        count = await chunks_repo.store_chunks("documents", doc_id, chunk_data)

        return Output(
            success=True,
            item={"doc_id": doc_id, "chunk_count": count, "already_existed": False},
        )

    async def _check_chunks(self, doc_id: str) -> Output:
        if not doc_id:
            return Output(success=False, error="--doc_id required")

        from app.db.repos.embedding_chunks import EmbeddingChunksRepo

        count = await EmbeddingChunksRepo().count_for_source("documents", doc_id)
        return Output(
            success=True,
            item={"doc_id": doc_id, "has_chunks": count > 0, "chunk_count": count},
        )


if __name__ == "__main__":
    DocumentManagerTool.run()
