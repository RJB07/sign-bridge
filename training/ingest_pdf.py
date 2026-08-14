#!/usr/bin/env python3
"""
ingest_pdf.py

Convert a sign-language PDF book into page images and extract page text.
Saves page images to `training/pdf_images/` and a JSON index `training/pdf_index.json`.

Dependencies: pdf2image, pillow, pytesseract

Usage:
  python training/ingest_pdf.py /path/to/book.pdf

"""
import sys
import os
import json
from pathlib import Path

try:
    from pdf2image import convert_from_path
    from PIL import Image
    import pytesseract
except Exception as e:
    print("Missing dependencies. Install with: pip install pdf2image pillow pytesseract")
    raise


OUT_DIR = Path(__file__).parent / 'pdf_images'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ingest(pdf_path: str):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"[*] Converting PDF pages to images: {pdf_path}")
    pages = convert_from_path(str(pdf_path), dpi=200)
    index = []
    for i, page in enumerate(pages, start=1):
        out_file = OUT_DIR / f"page_{i:04d}.png"
        page.save(out_file, 'PNG')
        print(f"[+] Saved page image: {out_file}")

        # Attempt OCR to pull any captions or labels
        try:
            text = pytesseract.image_to_string(page)
            text = text.strip()
        except Exception:
            text = ''

        index.append({
            'page': i,
            'image': str(out_file),
            'text': text,
        })

    idx_path = Path(__file__).parent / 'pdf_index.json'
    with open(idx_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"[*] Wrote PDF index to {idx_path} (pages: {len(index)})")

    # Generate repository training knowledge metadata file
    meta_path = Path(__file__).parent.parent / 'resources' / 'asl_books' / 'ingest_summary.json'
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
          'source_pdf': str(pdf_path.name),
          'total_pages': len(index),
          'status': 'INGESTED_TO_REPO'
        }, f, indent=2)
    print(f"[*] Updated book metadata manifest at {meta_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python training/ingest_pdf.py /path/to/book.pdf")
        sys.exit(1)
    ingest(sys.argv[1])

