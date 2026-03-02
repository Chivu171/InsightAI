from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# 1️ Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2️ Load file
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 3️ Chunking
def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

# 4️ Build FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # Tim vector co khoang cach Euclidean nho nhat
    index.add(embeddings)
    return index

# 5️ Main
if __name__ == "__main__":
    text = load_text("data/sample.txt")
    chunks = chunk_text(text)

    print("Number of chunks:", len(chunks))

    # Tạo embedding
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    print("Embedding shape:", embeddings.shape)

    # Tạo FAISS index
    index = build_faiss_index(embeddings)

    print("FAISS index built. Total vectors:", index.ntotal)

    while True:
        query = input("\nEnter your question (type 'exit' to quit): ")

        if query.lower() == "exit":
            print("Goodbye 👋")
            break

        # Encode query
        query_vector = model.encode([query])
        query_vector = np.array(query_vector).astype("float32")

        k = 3  # 🔥 top 3
        distances, indices = index.search(query_vector, k)

        print("\nTop 3 relevant chunks:")
        print("-" * 50)

        for rank, i in enumerate(indices[0]):
            print(f"\nRank {rank+1}")
            print("Distance:", distances[0][rank])
            print("Chunk:")
            print(chunks[i])
            print("-" * 50)