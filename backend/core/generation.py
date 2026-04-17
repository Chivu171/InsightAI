import os
from langchain_core.messages import HumanMessage, SystemMessage

class Generator:
    def __init__(self, llm):
        self.llm = llm

    def rewrite_query_with_history(self, user_query: str, history: list) -> str:
        """Uses LLM to rewrite the user query into a standalone question based on history."""
        if not history:
            return user_query
            
        history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in history[-5:]])
        
        prompt = f"""
[Nhiệm vụ] 
Dựa vào lịch sử hội thoại dưới đây, hãy viết lại câu hỏi mới nhất của người dùng thành một câu hỏi độc lập (standalone question), đầy đủ ngữ cảnh để có thể dùng tìm kiếm trong tài liệu. 

[Lịch sử hội thoại]
{history_text}

[Câu hỏi mới nhất]
{user_query}

[Yêu cầu]
- Nếu câu hỏi mới đã đầy đủ ý nghĩa, giữ nguyên.
- Nếu câu hỏi mới dùng đại từ thay thế (nó, cái đó, ông ấy...) hoặc liên quan đến ý trước, hãy thay bằng danh từ cụ thể từ lịch sử.
- TRẢ VỀ DUY NHẤT CÂU HỎI ĐÃ VIẾT LẠI, KHÔNG GIẢI THÍCH GÌ THÊM.
"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            rewritten_query = response.content.strip()
            print(f"[QueryRewriter] Original: '{user_query}' -> Rewritten: '{rewritten_query}'")
            return rewritten_query
        except Exception as e:
            print(f"[QueryRewriter] Error: {e}")
            return user_query

    def generate_answer(self, user_query: str, docs: list, original_query: str = None):
        """Generates a contextual answer using the provided documents."""
        relevant_chunks = [doc.page_content for doc in docs]
        if not relevant_chunks:
            return "Không có thông tin này trong tài liệu của bạn.", []

        context = "\n---\n".join(relevant_chunks)
        
        # Use original query in prompt if provided to maintain user intent
        query_to_use = original_query or user_query
        
        prompt = f"""
[Vai trò]
Bạn là trợ lý AI thông minh hỗ trợ phân tích tài liệu (RAG).

[Ngữ cảnh từ tài liệu]
{context}

[Câu hỏi]
{query_to_use}

[Yêu cầu trả lời]
1. Trả lời bằng tiếng Việt, rõ ràng, trung thực dựa trên ngữ cảnh được cung cấp.
2. Nếu ngữ cảnh không có thông tin, hãy nói "Tôi không tìm thấy thông tin này trong tài liệu".
3. Trích dẫn thông tin chính xác, có thể giải thích thêm nếu cần để làm rõ ý.
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content, docs

    def generate_summary(self, user_query: str, docs: list):
        """Generates a comprehensive summary or comparison based on the provided documents."""
        context = "\n---\n".join([d.page_content for d in docs])
        prompt = f"""
[Nhiệm vụ] Tóm tắt hoặc so sánh thông tin từ các đoạn văn bản sau để trả lời câu hỏi của người dùng.
[Ngữ cảnh]
{context}
[Câu hỏi]
{user_query}
[Yêu cầu]
- Trình bày mạch lạc, có thể dùng bullet points.
- Tập trung vào các ý chính quan trọng nhất.
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content, docs
