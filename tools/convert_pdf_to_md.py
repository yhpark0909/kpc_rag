from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable


def normalize_markdown(text: str, title: str) -> str:
    """PDF 추출 텍스트를 규정형 Markdown으로 정규화한다."""
    lines = [line.rstrip() for line in text.splitlines()]
    out = [f"# {title}", ""]

    chapter_re = re.compile(r"^제\d+장")
    section_re = re.compile(r"^제\d+절")
    article_re = re.compile(r"^제\d+조\(")

    prev_blank = False
    for raw in lines:
        line = raw.strip()
        if not line:
            if not prev_blank:
                out.append("")
            prev_blank = True
            continue

        if chapter_re.match(line):
            out.extend([f"## {line}", ""])
        elif section_re.match(line):
            out.extend([f"### {line}", ""])
        elif article_re.match(line):
            out.append(f"#### {line}")
        else:
            out.append(line)

        prev_blank = False

    return "\n".join(out).strip() + "\n"


def extract_pdf_text(pdf_path: Path) -> str:
    """PyMuPDFLoader로 PDF를 읽어서 텍스트를 결합한다."""
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "langchain_community 또는 PyMuPDFLoader를 불러오지 못했습니다. "
            "`pip install langchain-community pymupdf` 후 재실행하세요."
        ) from e

    try:
        docs = PyMuPDFLoader(str(pdf_path)).load()
    except Exception as e:
        raise RuntimeError(
            "PDF 본문 추출에 실패했습니다. "
            "스캔본이면 OCR 도구가 필요할 수 있습니다."
        ) from e

    text = "\n\n".join(doc.page_content for doc in docs).strip()
    if not text:
        raise RuntimeError("추출된 텍스트가 비어 있습니다.")

    return text


def convert_one(pdf_path: Path, out_path: Path | None = None) -> Path:
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF 파일이 아닙니다: {pdf_path}")

    text = extract_pdf_text(pdf_path)
    title = pdf_path.stem.split("_", 1)[0]
    md = normalize_markdown(text, title)

    target = out_path or pdf_path.with_suffix(".md")
    target.write_text(md, encoding="utf-8")
    return target


def iter_pdfs(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() == ".pdf":
        yield path
    elif path.is_dir():
        for p in sorted(path.glob("*.pdf")):
            if p.is_file():
                yield p


def main() -> int:
    parser = argparse.ArgumentParser(description="사규 PDF를 Markdown으로 변환")
    parser.add_argument(
        "input_path",
        type=Path,
        help="입력 경로 (PDF 파일 또는 PDF들이 있는 디렉토리)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="출력 파일 경로 (단일 PDF 변환일 때만 사용)",
    )
    args = parser.parse_args()

    targets = list(iter_pdfs(args.input_path))
    if not targets:
        print(f"변환 대상 PDF를 찾지 못했습니다: {args.input_path}", file=sys.stderr)
        return 1

    failed = 0
    for pdf in targets:
        try:
            out = convert_one(pdf, args.out if len(targets) == 1 else None)
            print(f"✅ 변환 완료: {pdf.name} -> {out}")
        except Exception as e:
            failed += 1
            print(f"❌ 변환 실패: {pdf.name} ({e})", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
