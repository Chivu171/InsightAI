import os
import csv
import io
import docx2txt
from datetime import datetime, timezone
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, embeddings: HuggingFaceEmbeddings):
        self.embeddings = embeddings
        self.parent_splitter = SemanticChunker(self.embeddings)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    def extract_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            import pypdf
            try:
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    return "\n".join([page.extract_text() or "" for page in reader.pages])
            except Exception as e:
                print(f"Error extracting PDF: {e}")
                return ""
        elif ext == ".docx":
            try:
                return docx2txt.process(file_path)
            except Exception as e:
                print(f"Error extracting Docx: {e}")
                return ""
        elif ext == ".csv":
            try:
                output = []
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        output.append(" | ".join(row))
                return "\n".join(output)
            except Exception as e:
                print(f"Error extracting CSV: {e}")
                return ""
        elif ext in [".txt", ".md"]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                print(f"Error extracting Text: {e}")
                return ""
        return ""

    def extract_documents(self, upload_file) -> list[Document]:
        filename = upload_file.filename
        temp_path = os.path.join("uploads", filename)
        
        try:
            with open(temp_path, "wb") as buffer:
                import shutil
                shutil.copyfileobj(upload_file.file, buffer)
        except Exception as e:
            print(f"Error saving file: {e}")
            return []
            
        full_text = self.extract_text(temp_path)
        if not full_text.strip():
            return []
            
        doc = Document(
            page_content=full_text,
            metadata={
                "document_name": filename,
                "file_type": os.path.splitext(filename)[1].lower(),
                "uploaded_at": datetime.now().isoformat(),
                "document_id": f"doc_{int(datetime.now().timestamp())}"
            }
        )
        return [doc]

    def process_into_chunks(self, docs: list[Document]):
        print(f"[Processor] Semantic chunking {len(docs)} documents...")
        parent_docs = self.parent_splitter.split_documents(docs)
        
        all_chunks = []
        for i, parent_doc in enumerate(parent_docs):
            parent_doc.metadata["parent_index"] = i
            child_chunks = self.child_splitter.split_documents([parent_doc])
            for child in child_chunks:
                child.metadata["parent_chunk_id"] = i
                all_chunks.append(child)
        
        return self._attach_chunk_metadata(all_chunks)

    def _attach_chunk_metadata(self, chunks: list[Document]) -> list[Document]:
        for chunk_index, chunk in enumerate(chunks, start=1):
            metadata = dict(chunk.metadata)
            document_id = metadata.get("document_id", "uploaded_file")
            page = metadata.get("page")
            page_label = f"p{page}" if page is not None else "p0"
            metadata["chunk_index"] = chunk_index
            metadata["chunk_id"] = f"{document_id}_{page_label}_c{chunk_index}"
            chunk.metadata = metadata
        return chunks

    def create_blocks(self, all_chunks: list[Document], block_size_chunks: int = 15, overlap_chunks: int = 3):
        if not all_chunks:
            return []
        
        print(f"[Processor] Creating blocks from {len(all_chunks)} chunks...")
        blocks = []
        step = max(1, block_size_chunks - overlap_chunks)
        for i in range(0, len(all_chunks), step):
            window = all_chunks[i : i + block_size_chunks]
            if not window: break
                
            block_text = "\n\n".join([doc.page_content for doc in window])
            block_ids = [doc.metadata.get("chunk_id") for doc in window if doc.metadata.get("chunk_id")]
            
            block_doc = Document(
                page_content=block_text.strip(),
                metadata={
                    "block_id": f"block_{len(blocks)}",
                    "child_chunk_ids": block_ids,
                    "chunk_count": len(window)
                }
            )
            blocks.append(block_doc)
            if i + block_size_chunks >= len(all_chunks): break
                
        return blocks
