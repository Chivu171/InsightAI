import sys
import os
import pickle
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from pypdf import PdfReader

try:
    import langchain
    print(f"DEBUG: Langchain found at {langchain.__path__}")
    from langchain.retrievers import ParentDocumentRetriever
    print("DEBUG: ParentDocumentRetriever imported successfully")
except ImportError as e:
    print(f"DEBUG: ImportError during langchain setup: {e}")
    print(f"DEBUG: sys.path is {sys.path}")
    # Fallback or re-raise
    raise
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class RAGEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2", gemini_model="models/gemini-flash-latest"):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.llm = genai.GenerativeModel(gemini_model)
        else:
            self.llm = None
        
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None
        self.docstore = InMemoryStore()
        self.retriever = None
        self.parent_splitter = SemanticChunker(self.embeddings)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    def extract_text(self, file_obj):
        """Extracts text from a file object (PDF or Text)."""
        if hasattr(file_obj, "name") and file_obj.name.endswith(".pdf"):
            reader = PdfReader(file_obj)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:
            # Assume text/bytes
            content = file_obj.read()
            return content.decode("utf-8") if isinstance(content, bytes) else content

    def chunk_text(self, text, chunk_size=500, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def build_index(self, text):
        """Builds the index using ParentDocumentRetriever with SemanticChunker."""
        docs = [Document(page_content=text, metadata={"source": "uploaded_file"})]
        
        # Initialize FAISS with an empty index if not already present
        if self.vectorstore is None:
            # We need at least one document to initialize FAISS, or use a dummy
            dummy_text = "initialization"
            self.vectorstore = FAISS.from_texts([dummy_text], self.embeddings)
            # Remove the dummy document if possible or just proceed
            
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            parent_splitter=None, # We'll handle parent splitting manually with SemanticChunker
        )
        
        # Split using SemanticChunker first
        parent_docs = self.parent_splitter.split_documents(docs)
        self.retriever.add_documents(parent_docs)
        return self.vectorstore, parent_docs

    def save_index(self, folder_path="vector_db"):
        if self.vectorstore is None:
            return False
        os.makedirs(folder_path, exist_ok=True)
        self.vectorstore.save_local(folder_path)
        # Note: InMemoryStore is not easily serializable with pickle if it contains complex objects,
        # but for strings/Documents it should be fine.
        with open(os.path.join(folder_path, "docstore.pkl"), "wb") as f:
            pickle.dump(self.docstore, f)
        return True

    def load_index(self, folder_path="vector_db"):
        if os.path.exists(os.path.join(folder_path, "index.faiss")):
            self.vectorstore = FAISS.load_local(folder_path, self.embeddings, allow_dangerous_deserialization=True)
            docstore_path = os.path.join(folder_path, "docstore.pkl")
            if os.path.exists(docstore_path):
                with open(docstore_path, "rb") as f:
                    self.docstore = pickle.load(f)
            
            self.retriever = ParentDocumentRetriever(
                vectorstore=self.vectorstore,
                docstore=self.docstore,
                child_splitter=self.child_splitter,
                parent_splitter=None,
            )
            return True
        return False

    def query(self, user_query, k=3):
        if self.retriever is None:
            # Try loading if exists
            if not self.load_index():
                return "Chưa có dữ liệu. Vui lòng tải file và index trước.", []
        
        relevant_docs = self.retriever.invoke(user_query)
        relevant_chunks = [doc.page_content for doc in relevant_docs]
        
        # Generate
        if not self.llm:
            return "LLM chưa được cấu hình (thiếu API Key).", relevant_chunks
        
        context = "\n---\n".join(relevant_chunks)
        prompt = f"""
Bạn là một trợ lý AI thông minh. Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp dưới đây.
Nếu thông tin không có trong ngữ cảnh, hãy nói rằng bạn không biết, đừng tự bịa ra câu trả lời.
Hãy trình bày câu trả lời một cách rõ ràng, sử dụng markdown để định dạng (danh sách, in đậm, bảng, v.v.) để người dùng dễ đọc nhất.

Câu hỏi: {user_query}

Ngữ cảnh:
{context}

Trả lời:
"""
        response = self.llm.generate_content(prompt)
        return response.text, relevant_chunks

if __name__ == "__main__":
    # Simple CLI for testing
    rag = RAGEngine()
    if rag.load_index():
        print("Loaded existing index.")
    else:
        print("No index found. Please use the Streamlit app to index data.")
    
    while True:
        q = input("\nHỏi (hoặc 'exit'): ")
        if q.lower() == 'exit': break
        ans, sources = rag.query(q)
        print(f"\nAI: {ans}")