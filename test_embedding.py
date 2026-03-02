from sentence_transformers import SentenceTransformer

# Load model embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Test text
text = "This paper proposes a novel deep learning architecture for image classification."

# Generate embedding
embedding = model.encode(text)

print("Embedding shape:", embedding.shape)
print("First 5 values:", embedding[:5])