#!/usr/bin/env python3
"""Extract text from book.pdf using PyMuPDF.

Usage:
    python scripts/extract_chapter.py <start_page> <end_page> [output_path]

Pages are PyMuPDF page numbers (1-indexed) as listed in study_notes/INDEX.md.
"""
import sys
from pathlib import Path

import fitz

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = REPO_ROOT / "book.pdf"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/chapter.txt")

    if not PDF_PATH.exists():
        print(f"book.pdf not found at {PDF_PATH}")
        return 2

    doc = fitz.open(PDF_PATH)
    parts = []
    for i in range(start - 1, end):
        parts.append(f"\n\n=== p.{i + 1} ===\n")
        parts.append(doc[i].get_text())
    text = "".join(parts)
    out.write_text(text)
    print(f"Extracted {len(text)} chars from p.{start}-{end} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
