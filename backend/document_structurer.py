import os
import re
from typing import List, Dict, Optional
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

class SectionNode:
    def __init__(self, title: str, level: int, parent=None):
        self.title = title
        self.level = level
        self.parent = parent
        self.children: List['SectionNode'] = []
        self.content: List[str] = []
        self.summary: Optional[str] = None

    def get_path(self) -> str:
        if self.parent:
            return f"{self.parent.get_path()} > {self.title}"
        return self.title

    def to_indented_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        marker = "📂" if self.children else "📝"
        result = f"{prefix}{marker} {self.title} (Level {self.level})\n"
        for child in self.children:
            result += child.to_indented_string(indent + 1)
        return result

class DocumentStructurer:
    def __init__(self):
        # We initialize the converter. This might download models on first run.
        self.converter = DocumentConverter()

    def process_file(self, file_path: str) -> List[Document]:
        """
        Parses a file into a list of Documents, each representing a section
        with structural metadata.
        """
        print(f"[Structurer] Converting {file_path} with Docling...")
        result = self.converter.convert(file_path)
        md_content = result.document.export_to_markdown()
        
        filename = os.path.basename(file_path)
        return self.parse_markdown_to_docs(md_content, filename)

    def parse_markdown_to_docs(self, md_text: str, filename: str) -> List[Document]:
        lines = md_text.split("\n")
        root = SectionNode(title="Root", level=0)
        current_node = root
        
        # Regex for markdown headers: # Title, ## Subtitle, etc.
        header_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

        for line in lines:
            match = header_pattern.match(line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                
                # Find the correct parent for this new header level
                while current_node.level >= level and current_node.parent:
                    current_node = current_node.parent
                
                new_node = SectionNode(title=title, level=level, parent=current_node)
                current_node.children.append(new_node)
                current_node = new_node
            else:
                if line.strip():
                    current_node.content.append(line)
        return root

    def parse_markdown_to_docs(self, md_text: str, filename: str) -> List[Document]:
        root = self._parse_markdown_to_tree(md_text, filename)
        # Flatten the tree into LangChain Documents
        documents = []
        self._flatten_tree(root, documents, filename)
        return documents

    def _parse_markdown_to_tree(self, md_text: str, filename: str) -> SectionNode:
        lines = md_text.split("\n")
        root = SectionNode(title="Root", level=0)
        current_node = root
        
        # Regex for markdown headers: # Title, ## Subtitle, etc.
        header_pattern = re.compile(r"^(#{1,6})\s+(.*)$")
        # Regex for section numbering: 1, 1.1, 1.2.1, etc.
        numbering_pattern = re.compile(r"^(\d+(?:\.\d+)*)\s+")

        for line in lines:
            match = header_pattern.match(line)
            if match:
                markdown_level = len(match.group(1))
                title = match.group(2).strip()
                
                # Heuristic: If title starts with numbering like 1.2.1, use that for level
                num_match = numbering_pattern.match(title)
                if num_match:
                    # "1" -> Level 1, "1.1" -> Level 2, etc.
                    inferred_level = len(num_match.group(1).split("."))
                    level = inferred_level
                else:
                    # Fallback to markdown level, but normalize if it's the first header
                    # Many PDFs export titles as Level 2 (##)
                    if markdown_level > 1 and current_node == root:
                        level = 1
                    else:
                        level = markdown_level
                
                # Find the correct parent for this new header level
                while current_node.level >= level and current_node.parent:
                    current_node = current_node.parent
                
                new_node = SectionNode(title=title, level=level, parent=current_node)
                current_node.children.append(new_node)
                current_node = new_node
            else:
                if line.strip():
                    current_node.content.append(line)
        return root

    def _flatten_tree(self, node: SectionNode, documents: List[Document], filename: str):
        # We skip the root node itself if it has no content
        if node.level > 0:
            full_content = "\n".join(node.content).strip()
            if full_content or node.children:
                # Even if content is empty (just a heading), we index it for routing
                path = node.get_path()
                # Remove "Root > " from path if it exists
                if path.startswith("Root > "):
                    path = path[7:]
                
                metadata = {
                    "document_name": filename,
                    "section_title": node.title,
                    "section_level": node.level,
                    "section_path": path,
                    "is_header_only": not full_content
                }
                
                documents.append(Document(
                    page_content=f"Section: {path}\n\n{full_content}",
                    metadata=metadata
                )
            )

        for child in node.children:
            self._flatten_tree(child, documents, filename)

if __name__ == "__main__":
    # Quick test
    structurer = DocumentStructurer()
    # Assuming there's a pdf in the root
    test_pdf = "../2312.10997.pdf"
    if os.path.exists(test_pdf):
        docs = structurer.process_file(test_pdf)
        for d in docs[:5]:
            print(f"PATH: {d.metadata['section_path']}")
            print(f"CONTENT: {d.page_content[:100]}...")
            print("-" * 20)
