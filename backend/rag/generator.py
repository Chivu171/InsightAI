import os
from typing import Tuple, List

import requests
from datetime import timezone, datetime


def format_request_error(error: requests.RequestException) -> str:
    response = getattr(error, "response", None)
    if response is not None:
        body = response.text.strip().replace("\n", " ")
        if len(body) > 300:
            body = body[:300] + "..."
        return f"HTTP {response.status_code} - {body or response.reason}"
    return str(error)


def build_citations(docs) -> list[dict]:
    citations = []
    for doc in docs:
        metadata = doc.metadata or {}
        snippet = " ".join(doc.page_content.split())
        citations.append(
            {
                "document_id": metadata.get("document_id", "unknown_document"),
                "document_name": metadata.get("document_name", "Tai lieu khong ro ten"),
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id"),
                "chunk_index": metadata.get("chunk_index"),
                "file_type": metadata.get("file_type"),
                "uploaded_at": metadata.get("uploaded_at"),
                "section_path": metadata.get("section_path"),
                "section_title": metadata.get("section_title"),
                "snippet": snippet[:320],
            }
        )
    return citations


def generate_with_lmstudio(engine, prompt: str) -> str:
    endpoint = f"{engine.local_base_url}/chat/completions"
    if not engine.local_base_url.endswith("/v1"):
        endpoint = f"{engine.local_base_url}/v1/chat/completions"

    response = requests.post(
        endpoint,
        json={
            "model": engine.local_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": False,
        },
        headers={"Content-Type": "application/json"},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def generate_with_google(engine, prompt: str) -> str:
    engine.api_key = os.getenv("GOOGLE_API_KEY", engine.api_key)
    response = requests.post(
        (
            "https://generativelanguage.googleapis.com/v1/models/"
            f"{engine.google_model}:generateContent?key={engine.api_key}"
        ),
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    candidates = payload.get("candidates", [])
    if not candidates:
        return ""

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    texts = [part.get("text", "") for part in parts if part.get("text")]
    return "\n".join(texts).strip()


def generate_answer_from_docs(engine, user_query: str, docs) -> Tuple[str, List[dict]]:
    citations = build_citations(docs)
    relevant_chunks = [doc.page_content for doc in docs]

    if not relevant_chunks:
        return "Khong co trong tai lieu.", citations

    context = "\n---\n".join(relevant_chunks)
    prompt = f"""
[Vai trò]
Bạn là trợ lý AI hỗ trợ người dùng đọc hiểu, phân tích và tra cứu thông tin từ tài liệu.

[Bối cảnh]
Bạn đang trả lời dựa trên ngữ cảnh được truy xuất từ tài liệu bằng hệ thống RAG.
Tài liệu có thể là PDF, TXT, CSV hoặc dữ liệu văn bản/bảng biểu tương tự.
Ngữ cảnh được cung cấp có thể chưa đầy đủ, nên bạn phải ưu tiên thông tin có trong ngữ cảnh trước.

[Mục tiêu]
Giúp người dùng:
- hiểu đúng nội dung tài liệu
- xác định thông tin nào thực sự có trong tài liệu
- phân biệt giữa nội dung có căn cứ trong tài liệu và phần suy luận/mở rộng
- nhận được câu trả lời rõ ràng, hữu ích và đủ chiều sâu khi cần

[Nhiệm vụ]
Tập trung vào việc trả lời đúng ý, rõ ràng, dễ hiểu và bám sát ngữ cảnh.
Không cần theo một cấu trúc trả lời cố định.
Có thể trả lời ngắn hoặc dài tùy theo câu hỏi.
Khi phù hợp, có thể mở rộng thêm kiến thức nền, cách hiểu hoặc bối cảnh liên quan để câu trả lời hữu ích hơn.

[Ràng buộc]
- Ưu tiên tuyệt đối thông tin có trong ngữ cảnh được cung cấp.
- Không được bịa hoặc khẳng định điều mà ngữ cảnh không hỗ trợ.
- Nếu ngữ cảnh không đủ để trả lời trọn vẹn, phải nói rõ phần nào có thể trả lời và phần nào chưa đủ dữ liệu.
- Nếu thông tin không có trong tài liệu, phải nói rõ: "Không có trong tài liệu".
- Nếu có phần mở rộng ngoài tài liệu, phải ghi rõ đó là phần giải thích thêm hoặc kiến thức nền, không phải nội dung được trích trực tiếp từ tài liệu.
- Nếu câu hỏi liên quan đến bảng, CSV hoặc số liệu, phải bám sát hàng, cột, giá trị và điều kiện có trong ngữ cảnh.
- Ưu tiên sự rõ ràng, chính xác và hữu ích hơn văn phong màu mè hoặc dài dòng.


Câu hỏi: {user_query}

Ngữ cảnh:
{context}

---
CRITICAL INSTRUCTION: You MUST answer the user's question in Vietnamese.
Trả lời (in English):
"""

    try:
        answer = generate_with_lmstudio(engine, prompt)
    except requests.RequestException as local_error:
        if not engine.api_key:
            print(f"[Generate] Local LLM error and no Google API key available: {local_error}")
            return "Khong the ket noi LM Studio va chua cung cap Google API Key.", citations

        try:
            answer = generate_with_google(engine, prompt)
        except requests.RequestException as google_error:
            print(f"[Generate] Local LLM error: {local_error}")
            print(f"[Generate] Google API error: {google_error}")
            return (
                "Khong the ket noi Local LLM. Google API loi: "
                f"{format_request_error(google_error)}",
                citations,
            )

    return answer or "Khong co trong tai lieu.", citations


def rewrite_query_with_history(engine, user_query: str, history_pairs: list[tuple[str, str]]) -> str:
    """Rewrite a conversational query into a standalone query using recent session history."""
    if not history_pairs:
        return user_query

    recent_history = history_pairs[-5:]
    history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in recent_history])

    prompt = f"""Dựa vào lịch sử hội thoại dưới đây, hãy viết lại câu hỏi cuối cùng của người dùng thành một câu hỏi độc lập (standalone query) để có thể dùng tìm kiếm trong tài liệu. 
Lưu ý: Chỉ trả về câu hỏi đã viết lại, không giải thích gì thêm. Nếu câu hỏi đã rõ ràng, hãy giữ nguyên.

Lịch sử hội thoại:
{history_text}

Câu hỏi mới: {user_query}
Standalone Query:"""

    try:
        if engine.local_base_url and engine.local_model:
            rewritten = generate_with_lmstudio(engine, prompt)
        else:
            rewritten = generate_with_google(engine, prompt)
        rewritten = (rewritten or "").strip()
        if rewritten:
            print(f"[Memory] Rewritten Query: {rewritten}")
            return rewritten
        return user_query
    except Exception:
        return user_query
