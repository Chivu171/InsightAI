from typing import Tuple, List

import requests
from openai import APIConnectionError, APIError, BadRequestError, RateLimitError

from config import settings


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


def _extract_openai_content(message) -> str: # Dùng trong 2 phương thức generate
    content = getattr(message, "content", "") or ""
    if not content:
        content = getattr(message, "reasoning_content", "") or ""
    return content.strip()


def generate_with_openrouter(engine, prompt: str) -> str:
    if engine.openrouter_client is None:
        raise RuntimeError("OpenRouter API key is not configured.")

    response = engine.openrouter_client.chat.completions.create(
        model=engine.openrouter_model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return _extract_openai_content(response.choices[0].message)


def generate_with_google(engine, prompt: str) -> str:
    engine.api_key = settings.google_api_key or engine.api_key
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


def generate_text(engine, prompt: str) -> str: # Kết hợp của 2 phương thức generate with open router và generate with LMStudio
    """Generate text using OpenRouter first, then legacy local/Google fallbacks.""" # Máy phát text bằng LLM
    if engine.openrouter_client is not None:
        try:
            return generate_with_openrouter(engine, prompt)
        except RateLimitError:
            return (
                "OpenRouter dang rate-limit model mien phi "
                f"`{engine.openrouter_model_name}`. Thu lai sau vai phut hoac doi model khac."
            )
        except (BadRequestError, APIError, APIConnectionError) as openrouter_error:
            print(f"[Generate] OpenRouter error: {openrouter_error}")
        except Exception as openrouter_error:
            print(f"[Generate] OpenRouter unexpected error: {openrouter_error}")

    if engine.local_base_url and engine.local_model:
        try:
            return generate_with_lmstudio(engine, prompt)
        except requests.RequestException as local_error:
            print(f"[Generate] Local LLM error: {local_error}")
    return "Chua cau hinh LLM provider. Vui long them OPENROUTER_API_KEY vao backend/.env."


def generate_answer_from_docs(engine, user_query: str, docs) -> Tuple[str, List[dict]]: #người soạn đề bài RAG rồi đưa cho cái máy phát text
    citations = build_citations(docs)
    relevant_chunks = [doc.page_content for doc in docs]

    if not relevant_chunks:
        return "Khong co trong tai lieu.", citations

    context = "\n---\n".join(relevant_chunks)
    prompt = f"""
[Vai trò]
Bạn là trợ lý AI chuyên hỗ trợ đọc hiểu bài báo khoa học, paper học thuật, báo cáo nghiên cứu và tài liệu kỹ thuật.

[Bối cảnh]
Bạn đang trả lời dựa trên ngữ cảnh được truy xuất từ tài liệu bằng hệ thống RAG.
Ngữ cảnh có thể là abstract, introduction, related work, methodology, experiments, results, discussion, conclusion, bảng biểu, hoặc caption hình.
Ngữ cảnh có thể chưa đầy đủ, vì vậy phải ưu tiên thông tin có trong ngữ cảnh trước và không tự suy diễn vượt quá dữ liệu.

[Mục tiêu]
Giúp người dùng:
- nắm được đề tài, câu hỏi nghiên cứu, và đóng góp chính của bài
- hiểu phương pháp, mô hình, dữ liệu, thiết lập thí nghiệm, metric và kết quả
- trích đúng insight học thuật từ bài báo
- phân biệt rõ nội dung có căn cứ trong paper với phần giải thích thêm
- đọc paper nhanh hơn nhưng vẫn chính xác

[Nhiệm vụ]
Trả lời theo đúng mục tiêu người dùng, có thể là:
- tóm tắt paper
- giải thích một đoạn, một công thức, một bảng, hoặc một hình
- so sánh với phương pháp khác
- chỉ ra điểm mạnh, điểm yếu, hạn chế, và giả định của nghiên cứu
- giải thích ý nghĩa thực nghiệm của kết quả

Nếu người dùng hỏi tóm tắt, hãy ưu tiên cấu trúc sau khi phù hợp:
1. Bài toán / mục tiêu nghiên cứu
2. Ý tưởng hoặc phương pháp chính
3. Dữ liệu / thiết lập / thực nghiệm
4. Kết quả chính
5. Hạn chế hoặc lưu ý

[Ràng buộc]
- Chỉ dùng thông tin có trong ngữ cảnh được cung cấp.
- Không bịa, không thêm kết luận mà paper không hỗ trợ.
- Nếu ngữ cảnh thiếu dữ liệu, phải nói rõ phần nào đủ để trả lời và phần nào chưa đủ.
- Nếu không tìm thấy trong tài liệu, phải nói rõ: "Không có trong tài liệu".
- Nếu cần giải thích thêm kiến thức nền, phải tách rõ đó là phần diễn giải thêm, không phải trích từ paper.
- Với bảng, số liệu, công thức, và metric, phải bám sát đúng giá trị trong ngữ cảnh.
- Nếu bài báo có thuật ngữ chuyên môn, ưu tiên giải thích dễ hiểu nhưng không làm sai nghĩa gốc.
- Trả lời bằng tiếng Việt, rõ ràng, chính xác, ngắn gọn khi câu hỏi đơn giản và chi tiết khi câu hỏi phức tạp.

Câu hỏi: {user_query}

Ngữ cảnh:
{context}

---
CHỈ DẪN QUAN TRỌNG: BẠN PHẢI trả lời câu hỏi của người dùng bằng tiếng Việt.
"""

    answer = generate_text(engine, prompt)

    return answer or "Khong co trong tai lieu.", citations


def rewrite_query_with_history(engine, user_query: str, history_pairs: list[tuple[str, str]]) -> str:
    """Rewrite a conversational query into a standalone query using recent session history."""
    if not history_pairs:
        return user_query

    recent_history = history_pairs[-5:]
    history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in recent_history])

    prompt = f"""Dựa vào lịch sử hội thoại dưới đây, hãy viết lại câu hỏi cuối cùng của người dùng thành một câu hỏi độc lập (standalone query) để có thể dùng tìm kiếm trong bài báo khoa học hoặc tài liệu nghiên cứu.
Lưu ý: Chỉ trả về câu hỏi đã viết lại, không giải thích gì thêm. Nếu câu hỏi đã rõ ràng, hãy giữ nguyên.

Lịch sử hội thoại:
{history_text}

Câu hỏi mới: {user_query}
Standalone Query:"""

    try:
        rewritten = generate_text(engine, prompt)
        rewritten = (rewritten or "").strip()
        if rewritten:
            print(f"[Memory] Rewritten Query: {rewritten}")
            return rewritten
        return user_query
    except Exception:
        return user_query
