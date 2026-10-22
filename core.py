"""Shared building blocks for this RAG demo.

Every repository in the *RAG Techniques* series ships an identical, self-contained
copy of this file. It deliberately keeps the LangChain surface small so the
technique script stays readable.

Nothing here is specific to a single technique: document loading, chunking,
the Chroma vector store, grounded answering, deduplication and reciprocal rank
fusion are reused everywhere.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PERSIST_DIR = ROOT / ".chroma"
COLLECTION = "rag_demo"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
DEFAULT_K = 4
RRF_K = 60
TEXT_SUFFIXES = {".txt", ".md"}

_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")

load_dotenv()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key, then run again."
        )


def chat_model(temperature: float = 0.0):
    """Return the chat model used by every technique in this repo."""
    from langchain_openai import ChatOpenAI

    _require_api_key()
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=temperature,
    )


def embedding_model():
    """Return the embedding model used by the vector stores."""
    from langchain_openai import OpenAIEmbeddings

    _require_api_key()
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def load_documents(data_dir: Path | str = DATA_DIR) -> list[Document]:
    """Load every .txt/.md file below *data_dir* as one Document per file."""
    data_dir = Path(data_dir)
    paths = sorted(
        p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
    )
    if not paths:
        raise SystemExit(
            f"No .txt or .md files found in {data_dir}.\n"
            "Put at least one document there (data/sample.txt is a good start)."
        )
    return [
        Document(
            page_content=p.read_text(encoding="utf-8"),
            metadata={"source": str(p.relative_to(data_dir))},
        )
        for p in paths
    ]


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks, keeping source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(list(documents))


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
def build_vectorstore(force: bool = True) -> Chroma:
    """Embed the corpus and persist a Chroma collection under ./.chroma."""
    if force and PERSIST_DIR.exists():
        import shutil

        shutil.rmtree(PERSIST_DIR)
    return Chroma.from_documents(
        documents=split_documents(load_documents()),
        embedding=embedding_model(),
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION,
    )


def get_vectorstore(force: bool = False) -> Chroma:
    """Open the persisted index, building it first if it is missing or empty."""
    if force or not PERSIST_DIR.exists() or not any(PERSIST_DIR.iterdir()):
        return build_vectorstore(force=True)
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embedding_model(),
    )


def in_memory_store(
    texts: Sequence[str],
    metadatas: Sequence[dict] | None = None,
    collection_name: str = "rag_demo_tmp",
) -> Chroma:
    """Build a throw-away in-memory Chroma store (used by index-rewriting recipes)."""
    return Chroma.from_texts(
        texts=list(texts),
        embedding=embedding_model(),
        metadatas=list(metadatas) if metadatas is not None else None,
        collection_name=collection_name,
    )


# --------------------------------------------------------------------------- #
# Grounded answering
# --------------------------------------------------------------------------- #
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Answer the question using only the numbered context passages. "
                "Cite the passages you rely on with their [n] markers. "
                "If the context does not contain the answer, reply exactly: "
                "I don't know based on the provided documents."
            ),
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

GENERAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a concise, helpful assistant. The user's question does not "
                "need the local document collection."
            ),
        ),
        ("human", "{question}"),
    ]
)


def format_context(documents: Sequence[Document]) -> str:
    """Render retrieved documents as numbered, source-tagged passages."""
    return "\n\n".join(
        f"[{i}] source={doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for i, doc in enumerate(documents, start=1)
    )


def answer(
    question: str,
    documents: Sequence[Document],
    model=None,
    *,
    show_sources: bool = False,
) -> str:
    """Produce an answer grounded in *documents*."""
    if not documents:
        return "I don't know based on the provided documents."
    chain = ANSWER_PROMPT | (model or chat_model()) | StrOutputParser()
    text = chain.invoke({"context": format_context(documents), "question": question}).strip()
    if show_sources:
        text += "\n\nSources:\n" + "\n".join(
            f"  [{i}] {doc.metadata.get('source', 'unknown')}"
            for i, doc in enumerate(documents, start=1)
        )
    return text


def plain_answer(question: str, model=None) -> str:
    """Answer without retrieval (used by the routing recipe)."""
    chain = GENERAL_PROMPT | (model or chat_model()) | StrOutputParser()
    return chain.invoke({"question": question}).strip()


# --------------------------------------------------------------------------- #
# Small utilities shared by the techniques
# --------------------------------------------------------------------------- #
def document_key(doc: Document) -> tuple[str, str]:
    """Stable identity for a chunk: source plus a content hash."""
    digest = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
    return str(doc.metadata.get("source", "")), digest


def dedupe(documents: Iterable[Document]) -> list[Document]:
    """Drop repeated chunks while preserving first-seen order."""
    seen: set[tuple[str, str]] = set()
    unique: list[Document] = []
    for doc in documents:
        key = document_key(doc)
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def strip_list_marker(line: str) -> str:
    """Remove a leading bullet or numbering marker from one generated line.

    Only the marker is removed, so a legitimate query such as
    ``"3 ways to chunk a document"`` keeps its leading number.
    """
    return _LIST_MARKER.sub("", line).strip()


def generated_lines(text: str) -> list[str]:
    """Split model output into cleaned, non-empty lines."""
    return [clean for line in text.splitlines() if (clean := strip_list_marker(line))]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    k: int = RRF_K,
) -> list[Document]:
    """Merge ranked lists with reciprocal rank fusion.

    Each document scores ``1 / (k + rank)`` in every list it appears in. Ties
    are broken by the stable document key so output order is reproducible.
    """
    scores: dict[tuple[str, str], float] = defaultdict(float)
    by_key: dict[tuple[str, str], Document] = {}
    for results in ranked_lists:
        for rank, doc in enumerate(results, start=1):
            key = document_key(doc)
            scores[key] += 1.0 / (k + rank)
            by_key.setdefault(key, doc)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [by_key[key] for key, _ in ordered]


def print_documents(documents: Sequence[Document], limit: int = 400) -> None:
    """Print retrieved chunks with their source and a truncated body."""
    for i, doc in enumerate(documents, start=1):
        body = doc.page_content.replace("\n", " ")
        print(f"\n[{i}] {doc.metadata}")
        print(body[:limit] + ("..." if len(body) > limit else ""))


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token) for cost/latency notes."""
    return max(1, len(text) // 4)


def question_from_args(
    argv: Sequence[str] | None = None,
    default: str = "What is retrieval-augmented generation?",
) -> str:
    """Read the question from argv, stdin, or a prompt, in that order."""
    argv = list(sys.argv if argv is None else argv)
    if len(argv) > 1:
        return " ".join(argv[1:]).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip() or default
    return input("Question: ").strip() or default
