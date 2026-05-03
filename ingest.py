"""Ingestão da base de conhecimento: leitura, chunking e indexação no ChromaDB."""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
from typing import Iterator, cast

import chromadb
from chromadb.api.types import Embeddable, EmbeddingFunction, Metadatas
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from docx import Document

from config import (
    DATA_DIR, DB_DIR, EMBED_MODEL, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXT,
)


# ---------- leitura por formato ----------

def read_pdf(path: Path) -> str:
    """Extrai texto de PDF página por página."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(f"[página {i + 1}]\n{text}")
    return "\n\n".join(pages)


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


READERS = {
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".md": read_text,
    ".txt": read_text,
}


# ---------- chunking ----------

def clean_text(text: str) -> str:
    """Normaliza espaços e quebras."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Chunking simples por caracteres com overlap, respeitando parágrafos quando possível."""
    text = clean_text(text)
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        # tenta cortar num final de frase/parágrafo próximo
        if end < n:
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > size * 0.5:  # só usa o corte se estiver razoavelmente longe do início
                end = start + cut + 1
        chunks.append(text[start:end].strip())
        start = end - overlap if end < n else end
    return [c for c in chunks if c]


# ---------- ingestão ----------

def file_id(path: Path) -> str:
    """ID estável baseado em caminho relativo + mtime, pra invalidar cache se o arquivo mudar."""
    rel = str(path.relative_to(DATA_DIR))
    h = hashlib.md5(f"{rel}:{path.stat().st_mtime}".encode()).hexdigest()[:12]
    return f"{rel}::{h}"


def iter_files(root: Path = DATA_DIR) -> Iterator[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            yield p


def get_collection():
    """Retorna a collection do ChromaDB com o embedder configurado."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    embedder = cast(
        EmbeddingFunction[Embeddable],
        embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_all(progress_cb=None) -> dict:
    """
    Indexa todos os arquivos suportados em DATA_DIR.
    progress_cb(current, total, filename) opcional para UI.
    Retorna estatísticas.
    """
    collection = get_collection()
    files = list(iter_files())
    stats = {"files": 0, "chunks": 0, "skipped": 0, "errors": []}

    # IDs já existentes pra evitar reprocessar (include=[] retorna só IDs, sem carregar embeddings)
    existing_ids = set(collection.get(include=[])["ids"])

    for idx, path in enumerate(files):
        if progress_cb:
            progress_cb(idx + 1, len(files), path.name)

        fid = file_id(path)
        rel_prefix = str(path.relative_to(DATA_DIR)) + "::"

        # chunks do mesmo arquivo (qualquer versão anterior)
        stale_ids = [_id for _id in existing_ids if _id.startswith(rel_prefix)]

        # se já tem chunks desta versão exata (mesmo mtime), pula
        if any(_id.startswith(fid) for _id in stale_ids):
            stats["skipped"] += 1
            continue

        # remove versões anteriores antes de reinserir
        if stale_ids:
            collection.delete(ids=stale_ids)
            for _id in stale_ids:
                existing_ids.discard(_id)

        reader = READERS.get(path.suffix.lower())
        if not reader:
            continue

        try:
            raw = reader(path)
        except Exception as e:
            stats["errors"].append(f"{path.name}: {e}")
            continue

        chunks = chunk_text(raw)
        if not chunks:
            continue

        ids = [f"{fid}::chunk{i}" for i in range(len(chunks))]
        metadatas: Metadatas = [
            {"source": str(path.relative_to(DATA_DIR)), "chunk_idx": i, "ext": path.suffix}
            for i in range(len(chunks))
        ]
        collection.add(ids=ids, documents=chunks, metadatas=metadatas)

        stats["files"] += 1
        stats["chunks"] += len(chunks)

    return stats


def reset_collection():
    """Apaga e recria a collection. Útil quando você quer reindexar do zero."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return get_collection()


def collection_stats() -> dict:
    col = get_collection()
    data = col.get()
    sources = {str(m["source"]) for m in data["metadatas"]} if data["metadatas"] else set()
    return {"chunks": len(data["ids"]), "sources": len(sources), "source_list": sorted(sources)}


if __name__ == "__main__":
    print("Ingerindo arquivos de", DATA_DIR)
    s = ingest_all(progress_cb=lambda i, t, n: print(f"  [{i}/{t}] {n}"))
    print("Concluído:", s)