import os
import sys
from rag_pipeline import RAGEngine

def verify():
    print("🚀 Initializing RAGEngine...")
    rag = RAGEngine()
    
    # Path to a sample PDF
    pdf_path = "../2312.10997.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: {pdf_path} not found. Please place a PDF in the root directory.")
        return

    print(f"📊 Building Structure-Aware Index for: {pdf_path}")
    rag.build_structure_index(pdf_path)
    
    query = "What are the core components of RAG according to this survey?"
    print(f"🔍 Querying: '{query}'")
    
    answer, citations = rag.query_with_structure(query)
    
    print("\n✨ ANSWER:")
    print(answer)
    
    print("\n📚 CITATIONS (Section-Aware):")
    for i, cit in enumerate(citations[:3]):
        path = cit.get('section_path', 'Unknown Section')
        snippet = cit.get('snippet', '')[:150]
        print(f"{i+1}. [{path}]")
        print(f"   Snippet: {snippet}...")
        print()

if __name__ == "__main__":
    verify()
