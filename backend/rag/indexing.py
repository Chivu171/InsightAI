import os
import pickle
import re
from datetime import datetime, timezone

import numpy as np
from rank_bm25 import BM25Okapi

try:
    import fitz  # PyMuPDF
except Exception:  # optional in dev
    fitz = None

import csv
import io
import tempfile
import docx2txt

from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_classic.retrievers import ParentDocumentRetriever
except ImportError:
    from langchain.retrievers import ParentDocumentRetriever

# Structural parsing was removed (DocumentStructurer deleted).


def clear_index(engine):
    engine.vectorstore = None
    engine.docstore = InMemoryStore()
    engine.retriever = None
    engine.chunking_mode = None
    engine.bm25_index = None
    engine.bm25_corpus = []
    engine.all_chunks = []
    engine.blocks = []
    engine.block_vectorstore = None
    engine.status = "idle"
    engine.progress = 0


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


def extract_documents(engine, file_obj):
    filename = getattr(file_obj, "filename", None) or getattr(file_obj, "name", "uploaded_file")
    real_file = getattr(file_obj, "file", file_obj)
    document_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", os.path.splitext(filename)[0]).strip("_") or "uploaded_file"
    uploaded_at = datetime.now(timezone.utc).isoformat()
    extension = os.path.splitext(filename)[1].lower()

    if hasattr(real_file, "seek"):
        real_file.seek(0)

    def _extract_docx_text(raw_bytes: bytes) -> str:
        # docx2txt works on file paths; write to a temp file.
        try:
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
                tmp.write(raw_bytes)
                tmp.flush()
                return (docx2txt.process(tmp.name) or "").strip()
        except Exception:
            return ""

    if extension == ".pdf":
        if fitz is None:
            raise RuntimeError("Missing dependency: PyMuPDF (fitz). Install it to process PDFs.")

        pdf_bytes = real_file.read()
        if hasattr(real_file, "seek"):
            real_file.seek(0)

        page_documents = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
            total_pages = len(pdf_document)
            for page_index, page in enumerate(pdf_document, start=1):
                page_text = page.get_text("text").strip()
                if not page_text:
                    continue
                page_documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "document_id": document_id,
                            "document_name": filename,
                            "file_type": extension or "pdf",
                            "page": page_index,
                            "total_pages": total_pages,
                            "uploaded_at": uploaded_at,
                        },
                    )
                )
        return page_documents

    if extension == ".docx":
        raw = real_file.read()
        if hasattr(real_file, "seek"):
            real_file.seek(0)

        if not isinstance(raw, (bytes, bytearray)):
            raw = str(raw).encode("utf-8", errors="ignore")

        docx_text = _extract_docx_text(bytes(raw))
        if not docx_text:
            return []

        return [
            Document(
                page_content=docx_text,
                metadata={
                    "document_id": document_id,
                    "document_name": filename,
                    "file_type": "docx",
                    "page": None,
                    "total_pages": None,
                    "uploaded_at": uploaded_at,
                },
            )
        ]

    if extension == ".csv":
        raw = real_file.read()
        if hasattr(real_file, "seek"):
            real_file.seek(0)

        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        reader = csv.DictReader(io.StringIO(text))

        row_documents = []
        for row_index, row in enumerate(reader, start=1):
            row_text = " | ".join(f"{key}: {value}" for key, value in row.items())
            if not row_text.strip():
                continue
            row_documents.append(
                Document(
                    page_content=row_text,
                    metadata={
                        "document_id": document_id,
                        "document_name": filename,
                        "file_type": "csv",
                        "page": None,
                        "row": row_index,
                        "total_pages": None,
                        "uploaded_at": uploaded_at,
                    },
                )
            )
        return row_documents

    content = real_file.read()
    if isinstance(content, bytes):
        # If the payload is actually a DOCX (even if renamed .txt), extract it.
        if content.startswith(b"PK\x03\x04"):
            docx_text = _extract_docx_text(content)
            if docx_text:
                return [
                    Document(
                        page_content=docx_text,
                        metadata={
                            "document_id": document_id,
                            "document_name": filename,
                            "file_type": "docx",
                            "page": None,
                            "total_pages": None,
                            "uploaded_at": uploaded_at,
                        },
                    )
                ]

        # TXT fallback decode
        text = None
        for enc in ("utf-8-sig", "utf-8"):
            try:
                text = content.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = content.decode("latin-1", errors="replace")
    else:
        text = content
    text = (text or "").strip()
    if not text:
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "document_id": document_id,
                "document_name": filename,
                "file_type": extension or "text",
                "page": None,
                "total_pages": None,
                "uploaded_at": uploaded_at,
            },
        )
    ]


def normalize_input_documents(text_or_docs):
    if isinstance(text_or_docs, str):
        return [
            Document(
                page_content=text_or_docs,
                metadata={
                    "document_id": "uploaded_file",
                    "document_name": "uploaded_file",
                    "file_type": "text",
                    "page": None,
                    "total_pages": None,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        ]
    return text_or_docs


def attach_chunk_metadata(chunks):
    for chunk_index, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk.metadata)
        document_id = metadata.get("document_id", "uploaded_file")
        page = metadata.get("page")
        page_label = f"p{page}" if page is not None else "p0"
        metadata["chunk_index"] = chunk_index
        metadata["chunk_id"] = f"{document_id}_{page_label}_c{chunk_index}"
        chunk.metadata = metadata
    return chunks




def build_index(engine, text_or_docs, chunking_mode="semantic", chunk_size=600, chunk_overlap=120):
    engine.status = "processing"
    engine.progress = 10

    docs = normalize_input_documents(text_or_docs)
    if not docs:
        raise ValueError("No documents available for indexing.")

    engine.progress = 30
    if chunking_mode == "fixed":
         parent_docs = split_fixed_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        parent_docs = engine.parent_splitter.split_documents(docs)
        parent_docs = attach_chunk_metadata(parent_docs)

    engine.progress = 50
    if engine.vectorstore is None:
        engine.vectorstore = FAISS.from_documents([parent_docs[0]], engine.embeddings)
        remaining_docs = parent_docs[1:]
    else:
        remaining_docs = parent_docs

    engine.retriever = ParentDocumentRetriever(
        vectorstore=engine.vectorstore,
        docstore=engine.docstore,
        child_splitter=engine.child_splitter,
        parent_splitter=None,
    )

    engine.progress = 60
    if remaining_docs:
        engine.retriever.add_documents(remaining_docs)

    engine.all_chunks.extend(parent_docs)
    engine.bm25_corpus = [tokenize(doc.page_content) for doc in engine.all_chunks]
    engine.bm25_index = BM25Okapi(engine.bm25_corpus)
    engine.chunking_mode = chunking_mode
    engine.chunk_size = chunk_size
    engine.chunk_overlap = chunk_overlap


    create_blocks(engine)

    engine.progress = 100
    engine.status = "done"
    print(f"[Hybrid] Built BM25 index with {len(engine.all_chunks)} chunks")
    return engine.vectorstore, parent_docs

def split_fixed_documents(docs, chunk_size = 600, chunk_overlap = 120):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return attach_chunk_metadata(chunks)

def create_blocks(engine, block_size_chunks: int = 15, overlap_chunks: int = 3):
    """Groups chunks into larger blocks using a chunk-based sliding window (summary retrieval)."""
    if not engine.all_chunks:
        engine.blocks = []
        engine.block_vectorstore = None
        return

    print(f"[Blocks] Creating blocks from {len(engine.all_chunks)} chunks (Sliding Window)...")
    blocks = []
    step = max(1, block_size_chunks - overlap_chunks)
    for i in range(0, len(engine.all_chunks), step):
        window = engine.all_chunks[i : i + block_size_chunks]
        if not window:
            break

        block_text = "\n\n".join([doc.page_content for doc in window])
        block_ids = [doc.metadata.get("chunk_id") for doc in window if doc.metadata.get("chunk_id")]

        block_doc = Document(
            page_content=block_text.strip(),
            metadata={
                "block_id": f"block_{len(blocks)}",
                "child_chunk_ids": block_ids,
                "chunk_count": len(window),
            },
        )
        blocks.append(block_doc)

        if i + block_size_chunks >= len(engine.all_chunks):
            break

    engine.blocks = blocks
    engine.block_vectorstore = FAISS.from_documents(blocks, engine.embeddings)
    print(f"[Blocks] Successfully built {len(blocks)} blocks.")


def save_index(engine, folder_path="vector_db"):
    if engine.vectorstore is None:
        return False

    os.makedirs(folder_path, exist_ok=True)
    engine.vectorstore.save_local(folder_path)
    with open(os.path.join(folder_path, "docstore.pkl"), "wb") as f:
        pickle.dump(engine.docstore, f)
    with open(os.path.join(folder_path, "bm25_data.pkl"), "wb") as f:
        pickle.dump({"all_chunks": engine.all_chunks, "bm25_corpus": engine.bm25_corpus}, f)

    with open(os.path.join(folder_path, "config.pkl"), "wb") as f:
        pickle.dump(
            {
                "chunking_mode": engine.chunking_mode,
                "chunk_size": getattr(engine, "chunk_size", None),
                "chunk_overlap": getattr(engine, "chunk_overlap", None),
            },
            f,
    )


    return True


def load_index(engine, folder_path="vector_db"):
    if not os.path.exists(os.path.join(folder_path, "index.faiss")):
        return False

    try:
        engine.vectorstore = FAISS.load_local(
            folder_path, engine.embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"[Load] FAISS load error: {e}")
        return False

    docstore_path = os.path.join(folder_path, "docstore.pkl")
    if os.path.exists(docstore_path):
        with open(docstore_path, "rb") as f:
            engine.docstore = pickle.load(f)

    engine.retriever = ParentDocumentRetriever(
        vectorstore=engine.vectorstore,
        docstore=engine.docstore,
        child_splitter=engine.child_splitter,
        parent_splitter=None,
    )

    bm25_path = os.path.join(folder_path, "bm25_data.pkl")
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        engine.all_chunks = data["all_chunks"]
        engine.bm25_corpus = data["bm25_corpus"]
        engine.bm25_index = BM25Okapi(engine.bm25_corpus)

    create_blocks(engine)

    config_path = os.path.join(folder_path, "config.pkl")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = pickle.load(f)
            engine.chunking_mode = config.get("chunking_mode", "semantic")
            engine.chunk_size = config.get("chunk_size", None)
            engine.chunk_overlap = config.get("chunk_overlap", None)

    else:
        engine.chunking_mode = "semantic"

    print(f"[Load] Index loaded with mode: {engine.chunking_mode}")
    return True
