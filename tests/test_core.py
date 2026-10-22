"""Offline tests for core.py.

These run without an API key: they cover the pure helpers plus a Chroma
round-trip backed by a deterministic fake embedding model.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

import core


def _doc(text: str, source: str = "a.txt") -> Document:
    return Document(page_content=text, metadata={"source": source})


def test_load_documents_finds_sample():
    docs = core.load_documents()
    assert docs
    assert all(d.page_content.strip() for d in docs)
    assert all("source" in d.metadata for d in docs)


def test_split_documents_preserves_source_and_start_index():
    docs = core.load_documents()
    chunks = core.split_documents(docs, chunk_size=400, chunk_overlap=50)
    assert len(chunks) > len(docs)
    assert {c.metadata["source"] for c in chunks} == {d.metadata["source"] for d in docs}
    assert all("start_index" in c.metadata for c in chunks)


def test_strip_list_marker_only_removes_markers():
    assert core.strip_list_marker("- first variant") == "first variant"
    assert core.strip_list_marker("* second") == "second"
    assert core.strip_list_marker("1. numbered") == "numbered"
    assert core.strip_list_marker("2) also numbered") == "also numbered"
    assert core.strip_list_marker("  \u2022 bullet") == "bullet"
    # A leading number that is part of the query must survive.
    assert core.strip_list_marker("3 ways to chunk a document") == "3 ways to chunk a document"
    assert core.strip_list_marker("2024 revenue") == "2024 revenue"


def test_generated_lines_drops_blanks_and_markers():
    text = "- alpha\n\n1. beta\n  \n* gamma\n"
    assert core.generated_lines(text) == ["alpha", "beta", "gamma"]


def test_dedupe_preserves_first_seen_order():
    docs = [_doc("one"), _doc("two"), _doc("one"), _doc("two")]
    unique = core.dedupe(docs)
    assert [d.page_content for d in unique] == ["one", "two"]


def test_dedupe_treats_identical_text_from_other_source_as_distinct():
    docs = [_doc("one"), _doc("one", "b.txt")]
    assert len(core.dedupe(docs)) == 2


def test_reciprocal_rank_fusion_orders_by_combined_score():
    a, b, c = _doc("alpha"), _doc("beta"), _doc("gamma")
    fused = core.reciprocal_rank_fusion([[a, b], [b, c]])
    assert fused[0] is b
    assert {id(d) for d in fused} == {id(a), id(b), id(c)}


def test_reciprocal_rank_fusion_is_deterministic():
    a, b = _doc("alpha"), _doc("beta")
    first = [core.document_key(d) for d in core.reciprocal_rank_fusion([[a, b], [b, a]])]
    second = [core.document_key(d) for d in core.reciprocal_rank_fusion([[a, b], [b, a]])]
    assert first == second


def test_format_context_is_numbered_and_tagged():
    rendered = core.format_context([_doc("hello", "x.txt"), _doc("world", "y.txt")])
    assert "[1] source=x.txt" in rendered
    assert "[2] source=y.txt" in rendered
    assert rendered.index("[1]") < rendered.index("[2]")


def test_estimate_tokens_is_positive():
    assert core.estimate_tokens("") == 1
    assert core.estimate_tokens("a" * 400) == 100


def test_in_memory_store_round_trip(monkeypatch):
    monkeypatch.setattr(core, "embedding_model", lambda: FakeEmbeddings(size=64))
    store = core.in_memory_store(
        texts=["retrieval finds relevant chunks", "generation grounds the answer"],
        metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
        collection_name="test_round_trip",
    )
    hits = store.similarity_search("retrieval", k=1)
    assert len(hits) == 1
    assert hits[0].metadata["source"] in {"a.txt", "b.txt"}
