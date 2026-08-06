import os
import pickle
import re
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mode as stat_mode

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

# ── Document-type detection ────────────────────────────────────────────────────

_ACADEMIC_KEYWORDS = [
    "abstract", "introduction", "methodology", "related work",
    "references", "conclusion", "experiments", "dataset", "baseline",
    "arxiv", "preprint", "proceedings", "journal", "doi", "figure",
    "table", "hypothesis", "empirical", "evaluation", "benchmark",
]

_LEGAL_KEYWORDS = [
    "whereas", "hereinafter", "clause", "agreement", "party",
    "indemnify", "jurisdiction", "liability", "pursuant", "obligation",
    "contract", "termination", "warranty", "indemnification",
]

# Canonical section names found in academic papers
_ACADEMIC_SECTIONS = {
    "abstract", "introduction", "related work", "background",
    "literature review", "methodology", "method", "methods",
    "approach", "model", "proposed method", "framework",
    "experiments", "experiment", "experimental setup", "experimental results",
    "results", "evaluation", "analysis", "discussion",
    "conclusion", "conclusions", "future work",
    "references", "appendix", "acknowledgment", "acknowledgements",
    "limitations", "ethics",
}


def detect_doc_type(docs: list) -> str:
    """Infer document type from the first ~2000 chars of content."""
    sample = " ".join(d.page_content for d in docs[:4]).lower()[:2000]
    academic_score = sum(1 for kw in _ACADEMIC_KEYWORDS if kw in sample)
    legal_score = sum(1 for kw in _LEGAL_KEYWORDS if kw in sample)
    if academic_score >= 3:
        return "academic_paper"
    if legal_score >= 3:
        return "legal"
    return "general"


def _extract_pdf_with_sections(
    pdf_bytes: bytes,
    document_id: str,
    filename: str,
    extension: str,
    uploaded_at: str,
) -> list:
    """
    Extract PDF pages as Documents, enriching each with section_title metadata.
    Uses font-size heuristic: spans significantly larger than body text are headings.
    Falls back to plain text extraction if font info is unavailable.
    """
    page_documents = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
        total_pages = len(pdf_document)

        # ── Pass 1: collect all font sizes to find body size ──────────────────
        all_sizes = []
        for page in pdf_document:
            try:
                blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
                for block in blocks:
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            sz = span.get("size", 0)
                            if sz > 0:
                                all_sizes.append(round(sz, 1))
            except Exception:
                pass

        # Body size = most frequent font size; heading threshold = 15% larger
        try:
            body_size = stat_mode(all_sizes) if all_sizes else 10.0
        except Exception:
            body_size = 10.0
        heading_threshold = body_size * 1.15

        # ── Pass 2: extract text + detect section headings ────────────────────
        current_section = "Unknown"

        for page_index, page in enumerate(pdf_document, start=1):
            try:
                blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            except Exception:
                # Fallback: plain text for this page
                page_text = page.get_text("text").strip()
                if page_text:
                    page_documents.append(Document(
                        page_content=page_text,
                        metadata={
                            "document_id": document_id,
                            "document_name": filename,
                            "file_type": extension or "pdf",
                            "page": page_index,
                            "total_pages": total_pages,
                            "uploaded_at": uploaded_at,
                            "section_title": current_section,
                        },
                    ))
                continue

            page_lines = []
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_text_parts = []
                    is_heading_line = False
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        size = span.get("size", 0)
                        if not text:
                            continue
                        line_text_parts.append(text)
                        # A line is a heading if any span is large enough
                        if size >= heading_threshold:
                            is_heading_line = True

                    line_text = " ".join(line_text_parts).strip()
                    if not line_text:
                        continue

                    # Check if heading matches known academic section names
                    if is_heading_line and len(line_text) < 80:
                        normalized = re.sub(r"^\d+[\.\s]+", "", line_text).strip().lower()
                        if any(sec in normalized for sec in _ACADEMIC_SECTIONS):
                            current_section = line_text.strip()

                    page_lines.append(line_text)

            page_text = "\n".join(page_lines).strip()
            if not page_text:
                continue

            page_documents.append(Document(
                page_content=page_text,
                metadata={
                    "document_id": document_id,
                    "document_name": filename,
                    "file_type": extension or "pdf",
                    "page": page_index,
                    "total_pages": total_pages,
                    "uploaded_at": uploaded_at,
                    "section_title": current_section,
                },
            ))

    return page_documents


def clear_index(engine):
    engine.vectorstore = None
    engine.docstore = InMemoryStore()
    engine.retriever = None
    engine.chunking_mode = None
    engine.doc_type = "general"
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

        return _extract_pdf_with_sections(
            pdf_bytes, document_id, filename, extension, uploaded_at
        )

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


def section_aware_split(docs: list, embeddings=None) -> list:
    """
    For academic papers: chunk within section boundaries.
    Uses RecursiveCharacterTextSplitter (fast) — sections already provide
    semantic context so SemanticChunker is redundant and very slow here.
    """
    # Group pages by section, preserving order
    section_order = []
    section_groups: dict[str, list] = defaultdict(list)

    for doc in docs:
        section = doc.metadata.get("section_title", "Unknown")
        if section not in section_groups:
            section_order.append(section)
        section_groups[section].append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks = []

    for section in section_order:
        section_docs = section_groups[section]
        chunks = splitter.split_documents(section_docs)

        # Propagate section_title to every child chunk
        for chunk in chunks:
            chunk.metadata["section_title"] = section

        all_chunks.extend(chunks)

    return attach_chunk_metadata(all_chunks)




def build_index(engine, text_or_docs, chunking_mode="semantic", chunk_size=600, chunk_overlap=120):
    engine.status = "processing"
    engine.progress = 10

    docs = normalize_input_documents(text_or_docs)
    if not docs:
        raise ValueError("No documents available for indexing.")

    # ── Auto-detect document type and choose chunking strategy ───────────────
    engine.doc_type = detect_doc_type(docs)
    print(f"[Adaptive] Detected doc_type: {engine.doc_type}")

    engine.progress = 30
    if chunking_mode == "fixed":
        parent_docs = split_fixed_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif chunking_mode == "semantic" and engine.doc_type == "academic_paper":
        # Academic paper: still detect sections via metadata, but use SemanticChunker
        print("[Adaptive] Using SemanticChunker for academic paper")
        parent_docs = engine.parent_splitter.split_documents(docs)
        parent_docs = attach_chunk_metadata(parent_docs)
    elif chunking_mode == "semantic":
        # General/legal doc: SemanticChunker for best quality (slower)
        print(f"[Adaptive] Using SemanticChunker for doc_type={engine.doc_type}")
        parent_docs = engine.parent_splitter.split_documents(docs)
        parent_docs = attach_chunk_metadata(parent_docs)
    elif chunking_mode == "semantic_only" and engine.doc_type == "academic_paper":
        # semantic_only + academic paper → section-aware (fast)
        print("[Adaptive] semantic_only + academic_paper → section-aware chunking")
        parent_docs = section_aware_split(docs)
    else:
        # semantic_only + general/legal: SemanticChunker (slow, opt-in explicitly)
        parent_docs = engine.parent_splitter.split_documents(docs)
        parent_docs = attach_chunk_metadata(parent_docs)

    engine.progress = 50
    if engine.vectorstore is None:
        engine.vectorstore = FAISS.from_documents([parent_docs[0]], engine.embeddings)
        remaining_docs = parent_docs[1:]
    else:
        remaining_docs = parent_docs

    if chunking_mode in ["fixed", "semantic_only"]:
        if remaining_docs:
            engine.vectorstore.add_documents(remaining_docs)
        engine.retriever = engine.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    else:
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
                "doc_type": getattr(engine, "doc_type", "general"),
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

    chunking_mode = "semantic"
    config_path = os.path.join(folder_path, "config.pkl")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = pickle.load(f)
            engine.chunking_mode = config.get("chunking_mode", "semantic")
            engine.chunk_size = config.get("chunk_size", None)
            engine.chunk_overlap = config.get("chunk_overlap", None)
            engine.doc_type = config.get("doc_type", "general")
            chunking_mode = engine.chunking_mode
    else:
        engine.chunking_mode = "semantic"

    docstore_path = os.path.join(folder_path, "docstore.pkl")
    if os.path.exists(docstore_path):
        with open(docstore_path, "rb") as f:
            engine.docstore = pickle.load(f)

    if chunking_mode in ["fixed", "semantic_only"]:
        engine.retriever = engine.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    else:
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

    print(f"[Load] Index loaded with mode: {engine.chunking_mode}")
    return True
