# InsightAI 🤖

InsightAI is a Retrieval-Augmented Generation (RAG) system designed to help researchers and students interact with their academic documents.

## 🚀 Features
- **Semantic Search**: Uses `sentence-transformers` to find the most relevant context.
- **Efficient Indexing**: Powered by FAISS for lightning-fast retrieval.
- **Intelligent Answers**: (In Progress) Connects to LLMs to generate grounded responses.
- **Easy Interface**: (Planned) Simple Web UI for document interaction.

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB**: `faiss-cpu`
- **LLM**: (TBD) OpenAI / Local Llama

## 📦 Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Chivu171/InsightAI.git
   cd InsightAI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline**:
   ```bash
   python rag_pipeline.py
   ```

## 📖 How it Works
1. **Ingestion**: Reads text files from the `data/` folder.
2. **Chunking**: Splits large texts into 200-character segments with 50-character overlap.
3. **Embedding**: Converts text into numerical vectors.
4. **Retrieval**: Finds the most similar chunks when you ask a question.
5. **Generation**: (Coming Soon) Uses the retrieved context to generate a final answer.

---
Built with ❤️ by Chivu171
