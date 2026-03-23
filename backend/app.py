import streamlit as st
import os
from rag_pipeline import RAGEngine

# Page config
st.set_page_config(page_title="InsightAI - Academic RAG", page_icon="🤖", layout="wide")

# Initialize RAG Engine in session state
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()
    st.session_state.rag.load_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Cấu hình")
    api_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        # Re-init if key changed
        if st.session_state.rag.api_key != api_key:
             st.session_state.rag = RAGEngine()
             st.session_state.rag.load_index()
    
    st.divider()
    st.subheader("📁 Dữ liệu")
    uploaded_file = st.file_uploader("Tải lên file PDF hoặc Text", type=["pdf", "txt"])
    
    if st.button("Re-index Data"):
        if uploaded_file:
            with st.spinner("Đang xử lý dữ liệu..."):
                text = st.session_state.rag.extract_text(uploaded_file)
                if text:
                    st.session_state.rag.build_index(text)
                    st.session_state.rag.save_index()
                    st.success(f"Đã index xong từ {uploaded_file.name}!")
                else:
                    st.error("Không thể đọc nội dung file.")
        else:
            st.warning("Vui lòng tải file lên trước.")

# Main UI
st.title("🤖 InsightAI")
st.markdown("Hỏi bất cứ điều gì về tài liệu của bạn.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Nhập câu hỏi tại đây..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if st.session_state.rag.vectorstore is None:
            st.warning("Vui lòng tải file và nhấn 'Re-index Data' ở sidebar trước.")
        else:
            with st.spinner("Đang tìm câu trả lời..."):
                answer, sources = st.session_state.rag.query(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Expander for sources
                if sources:
                    with st.expander("Xem nguồn trích dẫn"):
                        for i, chunk in enumerate(sources):
                            st.info(f"Nguồn {i+1}: {chunk}")
