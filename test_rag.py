import sys
import os

# Set up paths to import the backend code
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from rag_pipeline import RAGEngine

def test():
    print("Init RAGEngine...")
    rag = RAGEngine()
    
    print("Extract text 1...")
    text1 = "This is the first document about AI."
    print("Build index 1...")
    rag.build_index(text1)
    print("Save index 1...")
    rag.save_index()
    
    print("Reload index...")
    rag2 = RAGEngine()
    rag2.load_index()
    
    print("Extract text 2...")
    text2 = "This is the second document about Machine Learning."
    print("Build index 2 (Append)...")
    rag2.build_index(text2)
    print("Save index 2...")
    rag2.save_index()

    print("Success!")

if __name__ == "__main__":
    test()
