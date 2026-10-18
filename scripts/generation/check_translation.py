import argparse
import re
from pathlib import Path

from arxiv_rag import config
from arxiv_rag.generation import language, writer

SKIPPED = {"kaynakça", "ileri okuma", "references", "further reading"}


def split_sections(markdown: str) -> list[list[str]]:
    parts = re.split(r"(?m)^(## .*)$", markdown)
    return [[parts[0], ""]] + [[parts[i], parts[i + 1]] for i in range(1, len(parts), 2)]


def check(
    markdown: str, fix: bool = False, model: str = "", label: str = "literature survey"
) -> tuple[str, list[dict]]:
    sections, rows = split_sections(markdown), []
    for section in sections[1:]:
        heading, body = section
        name = heading.lstrip("#").strip()
        if name.lower().replace("\u0307", "") in SKIPPED:
            continue
        before = language.english_share(body)
        row = {"section": name, "before": before, "after": before, "repaired": 0, "failed": 0, "tokens": 0}
        if fix and before > writer.MAX_ENGLISH_SHARE:
            text, tokens, repaired, failed = writer.repair_paragraphs(body.strip(), "tr", model)
            section[1] = "\n\n" + text.strip() + "\n\n"
            row.update(after=language.english_share(text), repaired=repaired, failed=failed, tokens=tokens)
        rows.append(row)
    fixed = sections[0][0] + "".join(h + b for h, b in sections[1:])
    return fixed, rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Türkçe belgede İngilizce kalan bölümleri ölçer, istenirse yeniden çevirir."
    )
    p.add_argument("markdown", type=Path)
    p.add_argument("--fix", action="store_true", help="İngilizce kalan bölümleri yeniden çevir (LLM_API_KEY gerekir)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    text = args.markdown.read_text(encoding="utf-8")
    fixed, rows = check(text, args.fix, config.LLM_WRITE_MODEL)
    print(f"{'bölüm':<52}{'önce':>7}{'sonra':>7}{'onarılan':>10}{'başarısız':>11}{'token':>8}")
    for r in rows:
        print(
            f"{r['section'][:50]:<52}{r['before']:>7.0%}{r['after']:>7.0%}"
            f"{r['repaired']:>10}{r['failed']:>11}{r['tokens']:>8}"
        )
    body_before, body_after = (t.split("## Kaynakça")[0] for t in (text, fixed))
    print("\nBelge geneli önce:", language.audit(body_before))
    print("Belge geneli sonra:", language.audit(body_after))
    if args.fix:
        out = args.out or args.markdown.with_name(args.markdown.stem + "-duzeltilmis.md")
        out.write_text(fixed, encoding="utf-8")
        print(f"\nDüzeltilmiş belge: {out}")


if __name__ == "__main__":
    main()
