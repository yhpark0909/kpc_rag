from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable


def _remove_noise_lines(lines: list[str]) -> list[str]:
    """머리말/꼬리말/쪽번호 등 반복 노이즈를 제거한다."""
    cleaned: list[str] = []
    page_num_re = re.compile(r"^-?\s*\d+\s*-?$")
    footer_re = re.compile(r"^(인사관리\s+)?\d+-\d+$")

    for raw in lines:
        line = raw.strip()
        if not line:
            cleaned.append("")
            continue
        if page_num_re.match(line):
            continue
        if footer_re.match(line):
            continue
        cleaned.append(line)
    return cleaned


def _merge_wrapped_lines(lines: list[str]) -> list[str]:
    """PDF 줄바꿈으로 잘린 문장을 문맥 기준으로 병합한다."""
    merged: list[str] = []

    hard_start_re = re.compile(
        r"^(#{2,4}\s|제\d+장|제\d+절|제\d+조\(|[①-⑳]|\d+\.|[가-힣]\.|\*|-\s)"
    )

    for line in lines:
        if not line:
            if merged and merged[-1] != "":
                merged.append("")
            continue

        if not merged or merged[-1] == "" or hard_start_re.match(line):
            merged.append(line)
            continue

        # 직전 라인이 헤더면 본문은 새 줄에 둔다.
        if re.match(r"^#{2,4}\s", merged[-1]):
            merged.append(line)
            continue

        merged[-1] = f"{merged[-1]} {line}".replace("  ", " ").strip()

    return merged


def normalize_markdown(text: str, title: str) -> str:
    """PDF 추출 텍스트를 규정형 Markdown으로 정규화한다."""
    lines = [line.rstrip() for line in text.splitlines()]
    lines = _remove_noise_lines(lines)

    out = [f"# {title}", ""]
    chapter_re = re.compile(r"^(제\d+장[^\n]*)$")
    section_re = re.compile(r"^(제\d+절[^\n]*)$")
    article_re = re.compile(r"^(제\d+조\([^\n)]*\))\s*(.*)$")

    for raw in lines:
        line = raw.strip()
        if not line:
            out.append("")
            continue

        chapter_m = chapter_re.match(line)
        if chapter_m:
            out.extend([f"## {chapter_m.group(1)}", ""])
            continue

        section_m = section_re.match(line)
        if section_m:
            out.extend([f"### {section_m.group(1)}", ""])
            continue

        article_m = article_re.match(line)
        if article_m:
            out.append(f"#### {article_m.group(1)}")
            if article_m.group(2):
                out.append(article_m.group(2).strip())
            continue

        out.append(line)

    out = _merge_wrapped_lines(out)
    normalized = "\n".join(out)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized + "\n"


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
