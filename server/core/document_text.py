"""Text extraction for the formats the indexer accepts.

Uploads accept PDF, HWP, HWPX, DOCX and PPTX, and all five appear in
`rag_index_plan.INDEXABLE_EXTS`, but extraction only handled `.pdf` and
`.docx`; everything else fell through to
``read_text(encoding="utf-8", errors="ignore")`` and indexed mojibake.

Page numbers are never invented. The ``build_page_chunks`` contract is
``(page_num | None, text)``: PDF pages and PPTX slides really are 1-based
numbers, so they are carried; DOCX, HWPX and plain text have no such unit and
get ``None``.
"""

import logging
import re
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

#: ``(page_num | None, text)`` — the input contract of ``build_page_chunks``.
Page = Tuple[Optional[int], str]

#: Formats that genuinely have page or slide numbers. Everything else is ``None``.
PAGED_EXTS = frozenset({".pdf", ".pptx"})

#: Formats opened with a dedicated parser. Other indexable files are read as text.
BINARY_EXTS = frozenset({".pdf", ".docx", ".pptx", ".hwpx"})

#: Listed as indexable but not openable by this worker.
#: The values are reason codes; :func:`unsupported_reason` turns them into text.
UNSUPPORTED_EXTS = {".hwp": "needs_office_conversion"}

_REASON_EN = {
    "needs_office_conversion": (
        "{ext} cannot be opened by this worker (office conversion required) — skipped from indexing."
    ),
}

#: HWPX (OWPML) namespaces.
_HWPX_NS = {
    "hp": "http://www.hancom.co.kr/hwpml/2011/paragraph",
    "tbl": "http://www.hancom.co.kr/hwpml/2011/table",
}

#: A short paragraph without punctuation is treated as a heading.
_HWPX_HEADING_MAX_CHARS = 30


def is_unsupported(ext: str) -> bool:
    """Whether this extension cannot be opened here."""
    return (ext or "").lower() in UNSUPPORTED_EXTS


def unsupported_reason(ext: str, lang: object = None) -> str:
    """Why it cannot be opened. Failing silently reads as "why is search broken"."""

    code = UNSUPPORTED_EXTS.get((ext or "").lower(), "")
    table = _REASON_EN
    template = table.get(code)
    if not template:
        return ""
    return template.format(ext=(ext or "").lower())


# ══════════════════════════════════════════════════════════════════
# Per-format extraction
# ══════════════════════════════════════════════════════════════════


def _pdf_pages(filepath: Path) -> List[Page]:
    import pdfplumber

    with pdfplumber.open(filepath) as pdf:
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(pdf.pages)]


def _docx_pages(filepath: Path) -> List[Page]:
    from docx import Document

    doc = Document(str(filepath))
    text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(None, text)]


def _pptx_pages(filepath: Path) -> List[Page]:
    """One entry per slide, with the 1-based slide number as ``page_num``.

    Table cells are joined with spaces: the index only needs searchable text, so
    formatting is not restored. Speaker notes are excluded — they are not body text.
    """
    from pptx import Presentation

    prs = Presentation(str(filepath))
    pages: List[Page] = []
    for idx, slide in enumerate(prs.slides, start=1):
        parts: List[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                txt = shape.text_frame.text.strip()
                if txt:
                    parts.append(txt)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append(" ".join(cells))
        pages.append((idx, "\n".join(parts)))
    return pages


def hwpx_text(filepath: Path) -> str:
    """HWPX (zip plus OWPML XML) to plain text, using only the standard library.

    Paragraphs come out in order and tables row by row. Unlike the Markdown
    exporter this was ported from, no file scaffolding (headings, source lines)
    is produced: indexing only needs searchable text.
    """
    import xml.etree.ElementTree as ET

    def _text_of(elem) -> str:
        return "".join(elem.itertext()).strip()

    paras: List[str] = []
    tables: List[str] = []
    with zipfile.ZipFile(filepath, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        for name in sorted(names):
            with zf.open(name) as fp:
                try:
                    root = ET.parse(fp).getroot()
                except Exception:  # noqa: BLE001 — one broken XML part must not discard the document
                    continue
                for tbl in root.findall(".//tbl:tbl", _HWPX_NS):
                    for row in tbl.findall(".//tbl:tr", _HWPX_NS):
                        cells = []
                        for cell in row.findall(".//tbl:tc", _HWPX_NS):
                            cell_text = " ".join(_text_of(p) for p in cell.findall(".//hp:para", _HWPX_NS)).strip()
                            if cell_text:
                                cells.append(cell_text)
                        if cells:
                            tables.append(" ".join(cells))
                for para in root.findall(".//hp:para", _HWPX_NS):
                    txt = _text_of(para)
                    if not txt:
                        continue
                    if len(txt) <= _HWPX_HEADING_MAX_CHARS and not re.search(r"[.!?:]", txt):
                        paras.append(f"## {txt}")
                    else:
                        paras.append(txt)
    return "\n\n".join(paras + tables).strip()


def _hwpx_pages(filepath: Path) -> List[Page]:
    return [(None, hwpx_text(filepath))]


def _text_pages(filepath: Path) -> List[Page]:
    return [(None, filepath.read_text(encoding="utf-8", errors="ignore"))]


_EXTRACTORS = {
    ".pdf": _pdf_pages,
    ".docx": _docx_pages,
    ".pptx": _pptx_pages,
    ".hwpx": _hwpx_pages,
}


def extract_pages(filepath: Path, lang: object = None) -> List[Page]:
    """Extract a document into ``[(page_num | None, text)]``.

    **An unopenable document yields an empty list**, never a best-effort UTF-8
    read: broken strings in the index make search return garbage rather than
    nothing. The reason goes to the log.
    """
    ext = filepath.suffix.lower()

    if is_unsupported(ext):
        logger.warning("[document_text] %s — %s", filepath.name, unsupported_reason(ext, lang))
        return []

    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        return _text_pages(filepath)

    try:
        return extractor(filepath)
    except ImportError as exc:
        # Parser library missing — do not fall back to plain text; that was the original defect.
        logger.error("[document_text] could not load the %s parser (%s): %s", ext, filepath.name, exc)
        return []
    except Exception as exc:  # noqa: BLE001 — one document must not kill the whole indexing run
        logger.warning("[document_text] %s extraction failed (%s): %s", ext, filepath.name, exc)
        return []
