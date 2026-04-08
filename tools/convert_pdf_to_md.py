from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable


_WS_RE = re.compile(r"\s+")
_PAGE_ONLY_RE = re.compile(r"^\d+$")
_HEADER_FOOTER_RE = re.compile(r"^(?:-{2,}|_{2,}|…+)$")

_CHAPTER_RE = re.compile(r"^제\s*\d+\s*장(?:\s+.+)?$")
_SECTION_RE = re.compile(r"^제\s*\d+\s*절(?:\s+.+)?$")
_ARTICLE_LINE_RE = re.compile(r"^(제\s*\d+\s*조(?:\s*\([^)]+\))?)(?:\s+(.*))?$")
_ITEM_RE = re.compile(r"^(?:(?P<circled>[①-⑳])|(?P<number>\d{1,2})[.)])\s*(?P<body>.+)$")
_DOC_TITLE_RE = re.compile(r"^[가-힣A-Za-z0-9·\s\-\(\)]+규칙$|^[가-힣A-Za-z0-9·\s\-\(\)]+규정$")


def _clean_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = line.replace("­", "")  # soft hyphen
    line = _WS_RE.sub(" ", line).strip()
    return line


def _infer_title(text: str, fallback: str) -> str:
    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            continue
        if _DOC_TITLE_RE.match(line):
            return line
        if line.startswith("제") and ("장" in line or "조" in line):
            break
        # 규칙/규정 키워드가 없어도 첫 유효 라인을 제목 후보로 사용
        if len(line) <= 40:
            return line
    return fallback


def normalize_markdown(text: str, title: str) -> str:
    """PDF 추출 텍스트를 사규용 Markdown으로 정교하게 정규화한다."""
    out = [f"# {title}", ""]
    prev_blank = False

    for raw in text.splitlines():
        line = _clean_line(raw)
        if (
            not line
            or _PAGE_ONLY_RE.match(line)
            or _HEADER_FOOTER_RE.match(line)
        ):
            if not prev_blank:
                out.append("")
            prev_blank = True
            continue

        if _CHAPTER_RE.match(line):
            out.extend([f"## {line}", ""])
            prev_blank = False
            continue

        if _SECTION_RE.match(line):
            out.extend([f"### {line}", ""])
            prev_blank = False
            continue

        article_m = _ARTICLE_LINE_RE.match(line)
        if article_m:
            heading = article_m.group(1)
            tail = article_m.group(2)
            out.extend([f"#### {heading}", ""])
            if tail:
                out.append(tail)
            prev_blank = False
            continue

        item_m = _ITEM_RE.match(line)
        if item_m:
            marker = item_m.group("circled") or f"{item_m.group('number')}."
            body = item_m.group("body")
            out.append(f"- **{marker}** {body}")
            prev_blank = False
            continue

        out.append(line)
        prev_blank = False

    # 연속 공백 라인/끝 공백 정리
    normalized: list[str] = []
    for line in out:
        if line == "" and normalized and normalized[-1] == "":
            continue
        normalized.append(line)

    return "\n".join(normalized).strip() + "\n"


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
    title = _infer_title(text, pdf_path.stem.split("_", 1)[0])
    md = normalize_markdown(text, title)

    target = out_path or pdf_path.with_suffix(".md")
    target.write_text(md, encoding="utf-8")
    return target


def iter_pdfs(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() == ".pdf":
        yield path
    elif path.is_dir():
        for p in sorted(path.rglob("*.pdf")):
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
    parser.add_argument(
        "--keep-pdf",
        action="store_true",
        help="변환 후 원본 PDF를 삭제하지 않고 유지합니다.",
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
            if not args.keep_pdf:
                pdf.unlink(missing_ok=True)
            print(f"✅ 변환 완료: {pdf.name} -> {out}")
        except Exception as e:
            failed += 1
            print(f"❌ 변환 실패: {pdf.name} ({e})", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
