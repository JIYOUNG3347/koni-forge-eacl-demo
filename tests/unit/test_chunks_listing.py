"""Pure module — no fastapi/celery/chromadb. Covers page_num injection
(build_page_chunks) and the listing response shaping (build_chunks_listing):
new-format chunks expose page_num, legacy chunks fall back to None, plus
limit/offset paging and empty handling.
"""

from server.core.chunk_meta import build_chunks_listing, build_page_chunks


# ── build_page_chunks (page_num injection) ──────────────────────────────────
def test_page_chunks_pdf_preserves_page_num():
    pages = [(1, "Content of the first page."), (2, "Content of the second page.")]
    recs = build_page_chunks(pages, chunk_size=512, chunk_overlap=50)
    assert [r["page_num"] for r in recs] == [1, 2]
    assert [r["chunk_index"] for r in recs] == [0, 1]  # sequential across the document
    assert recs[0]["text"] == "Content of the first page."


def test_page_chunks_non_paged_is_none():
    recs = build_page_chunks([(None, "Markdown body. No page concept.")])
    assert len(recs) == 1
    assert recs[0]["page_num"] is None
    assert recs[0]["chunk_index"] == 0


def test_page_chunks_global_index_across_pages():
    long = "Sentence. " * 400  # so one page splits into several chunks
    recs = build_page_chunks([(1, long), (2, long)], chunk_size=512, chunk_overlap=50)
    # chunk_index runs 0..N-1 without a break at a page boundary.
    assert [r["chunk_index"] for r in recs] == list(range(len(recs)))
    assert recs[0]["page_num"] == 1
    assert recs[-1]["page_num"] == 2


def test_page_chunks_empty_text_skipped():
    recs = build_page_chunks([(1, ""), (2, "content")])
    assert [r["page_num"] for r in recs] == [2]


# ── build_chunks_listing (response shaping) ─────────────────────────────────
def _rec(i, page, text="body", doc="a.pdf"):
    r = {"chunk_index": i, "doc_name": doc, "text": text}
    if page is not None:
        r["page_num"] = page
    return r


def test_listing_new_format_includes_page_num():
    records = [_rec(0, 3), _rec(1, 4)]
    out = build_chunks_listing(records, limit=20, offset=0)
    assert out["total"] == 2
    assert out["chunks"][0]["page_num"] == 3
    assert out["chunks"][1]["page_num"] == 4
    assert out["chunks"][0]["doc_name"] == "a.pdf"


def test_listing_legacy_chunk_page_num_null():
    # Older chunks have no page_num key, so it reads as None.
    records = [{"chunk_index": 0, "doc_name": "old.pdf", "text": "x"}]
    out = build_chunks_listing(records)
    assert out["total"] == 1
    assert out["chunks"][0]["page_num"] is None


def test_listing_truncates_text():
    long = "x" * 500
    out = build_chunks_listing([_rec(0, 1, text=long)], preview_chars=200)
    assert len(out["chunks"][0]["text"]) == 200
    assert out["chunks"][0]["truncated"] is True


def test_listing_paging_limit_offset():
    records = [_rec(i, 1) for i in range(50)]
    page2 = build_chunks_listing(records, limit=20, offset=20)
    assert page2["total"] == 50
    assert len(page2["chunks"]) == 20
    assert page2["chunks"][0]["chunk_index"] == 20
    assert page2["chunks"][-1]["chunk_index"] == 39


def test_listing_sorts_by_chunk_index():
    records = [_rec(2, 1), _rec(0, 1), _rec(1, 1)]  # shuffled
    out = build_chunks_listing(records)
    assert [c["chunk_index"] for c in out["chunks"]] == [0, 1, 2]


def test_listing_empty_dataset():
    out = build_chunks_listing([])
    assert out == {"chunks": [], "total": 0}
