import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "backend"))

from rag_pipeline import RAGEngine  # noqa: E402


WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def extract_questions_from_docx(docx_path: Path) -> list[str]:
    questions: list[str] = []

    with zipfile.ZipFile(docx_path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ElementTree.fromstring(xml_bytes)
    for paragraph in root.findall(".//w:p", WORD_NAMESPACE):
        texts = [
            node.text.strip()
            for node in paragraph.findall(".//w:t", WORD_NAMESPACE)
            if node.text and node.text.strip()
        ]
        line = " ".join(texts).strip()
        if "👉" not in line:
            continue

        line = re.sub(r"^.*?👉\s*", "", line).strip()
        if line:
            questions.append(line)

    return questions[:10]


def load_documents(engine: RAGEngine, source_path: Path):
    with source_path.open("rb") as handle:
        return engine.extract_documents(handle)


def summarize_sources(sources: list[dict]) -> list[dict]:
    return [
        {
            "document_name": source.get("document_name"),
            "page": source.get("page"),
            "parent_id": source.get("parent_id"),
            "child_idx": source.get("child_idx"),
            "snippet": source.get("snippet"),
        }
        for source in sources
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to the source PDF/TXT used to build the semantic index.")
    parser.add_argument(
        "--questions-docx",
        default=str(ROOT / "🔹 10 câu hỏi test RAG.docx"),
        help="Path to the benchmark DOCX containing the 10 questions.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "benchmark_local_window_results.json"),
        help="Where to write the comparison results.",
    )
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    questions_docx = Path(args.questions_docx).resolve()
    output_path = Path(args.output).resolve()

    engine = RAGEngine()
    documents = load_documents(engine, source_path)
    engine.clear_index()
    engine.build_index(documents)

    questions = extract_questions_from_docx(questions_docx)
    results = []

    for question in questions:
        parent_answer, parent_sources = engine.query_full_parent(question)
        local_answer, local_sources = engine.query(question)
        results.append(
            {
                "question": question,
                "full_parent": {
                    "answer": parent_answer,
                    "sources": summarize_sources(parent_sources),
                },
                "local_child_window": {
                    "answer": local_answer,
                    "sources": summarize_sources(local_sources),
                },
            }
        )

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} benchmark comparisons to {output_path}")


if __name__ == "__main__":
    main()
