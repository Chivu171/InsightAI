import os
import pickle
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2", gemini_model="models/gemini-flash-latest"):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.llm = genai.GenerativeModel(gemini_model)
        else:
            self.llm = None
        
        self.embed_model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = None

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
        self.chunks = self.chunk_text(text)
        embeddings = self.embed_model.encode(self.chunks)
        embeddings = np.array(embeddings).astype("float32")
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        return self.index, self.chunks

    def save_index(self, index_path="vector_db/index.faiss", chunks_path="vector_db/chunks.pkl"):
        if self.index is None or self.chunks is None:
            return False
        os.makedirs("vector_db", exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        return True

    def load_index(self, index_path="vector_db/index.faiss", chunks_path="vector_db/chunks.pkl"):
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            return True
        return False

    def query(self, user_query, k=3):
        if self.index is None or self.chunks is None:
            return "Chưa có dữ liệu. Vui lòng tải file và index trước.", []
        
        # Search
        query_vector = self.embed_model.encode([user_query])
        query_vector = np.array(query_vector).astype("float32")
        distances, indices = self.index.search(query_vector, k)
        relevant_chunks = [self.chunks[i] for i in indices[0]]
        
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
        print(f"Loaded index with {rag.index.ntotal} vectors.")
    else:
        print("No index found. Please use the Streamlit app to index data.")
    
    while True:
        q = input("\nHỏi (hoặc 'exit'): ")
        if q.lower() == 'exit': break
        ans, sources = rag.query(q)
        print(f"\nAI: {ans}")