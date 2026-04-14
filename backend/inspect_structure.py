import os
import sys
from document_structurer import DocumentStructurer, SectionNode

def main():
    if len(sys.argv) < 2:
        # Default to the survey PDF if no argument provided
        pdf_path = "/Users/apple/Downloads/2203.11026v1.pdf"
    else:
        pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found: {pdf_path}")
        return

    print(f"🔍 Analyzing structure of: {pdf_path}...")
    structurer = DocumentStructurer()
    
    # We need to get the raw conversion first
    result = structurer.converter.convert(pdf_path)
    md_content = result.document.export_to_markdown()
    
    # Parse to tree
    root = structurer._parse_markdown_to_tree(md_content, os.path.basename(pdf_path))
    
    # Generate visualization
    tree_viz = root.to_indented_string()
    
    # Print to console
    print("\n" + "="*50)
    print("📄 DOCUMENT STRUCTURE (TOC)")
    print("="*50)
    print(tree_viz)
    print("="*50)
    
    # Save to file
    output_filename = f"structure_{os.path.splitext(os.path.basename(pdf_path))[0]}.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Document Structure for: {pdf_path}\n")
        f.write("="*50 + "\n")
        f.write(tree_viz)
    
    print(f"\n✅ Structure exported to: {output_filename}")

if __name__ == "__main__":
    main()
