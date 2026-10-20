import argparse
import re
from pathlib import Path

from arxiv_rag import config
from arxiv_rag.generation import language, writer

SKIPPED = {"kaynakça", "ileri okuma", "references", "further reading"}


TITLE_LINE = re.compile(r"^\[\d+\] .*?\((?:\d{4}|n\.d\.)\)\. (.+?)\.? arXiv:\S+\. https://arxiv\.org/abs/", re.M)
FURTHER_LINE = re.compile(r"^- (.+) \((?:\d{4}|n\.d\.)\)\. https://arxiv\.org/abs/", re.M)


def titles_in(markdown: str) -> list[str]:
    return TITLE_LINE.findall(markdown) + FURTHER_LINE.findall(markdown)


def english_left(text: str) -> int:
    found = language.audit(text)
    return found["english_paragraphs"] + found["english_tables"] + found["english_lines"]


def split_sections(markdown: str) -> list[list[str]]:
    parts = re.split(r"(?m)^(## .*)$", markdown)
    return [[parts[0], ""]] + [[parts[i], parts[i + 1]] for i in range(1, len(parts), 2)]


def check(
    markdown: str, fix: bool = False, model: str = "", label: str = "literature survey"
) -> tuple[str, list[dict]]:
    sections, rows, titles = split_sections(markdown), [], titles_in(markdown)
    for section in sections[1:]:
        heading, body = section
        name = heading.lstrip("#").strip()
        if name.lower().replace("\u0307", "") in SKIPPED:
            continue
        before = language.english_share(body)
        heading_english = language.text_is_english(name)
        row = {
            "section": name,
            "before": before,
            "after": before,
            "left_before": english_left(body),
            "left_after": english_left(body),
            "heading_english": heading_english,
            "repaired": 0,
            "failed": 0,
            "tokens": 0,
        }
        if fix and heading_english:
            translated, tokens = writer.translate_lines([name], "tr", model)
            section[0] = heading.replace(name, translated[0], 1)
            row.update(heading_english=language.text_is_english(translated[0]), tokens=tokens)
        if fix and language.has_english(body):
            text, tokens, repaired, failed = writer.repair_paragraphs(body.strip(), "tr", model, titles=titles)
            section[1] = "\n\n" + text.strip() + "\n\n"
            row.update(
                after=language.english_share(text),
                left_after=english_left(text),
                repaired=repaired,
                failed=failed,
                tokens=row["tokens"] + tokens,
            )
        rows.append(row)
    fixed = sections[0][0] + "".join(h + b for h, b in sections[1:])
    return fixed, rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Türkçe belgede İngilizce kalan paragrafları, tabloları ve başlıkları ölçer, istenirse yeniden çevirir."
    )
    p.add_argument("markdown", type=Path)
    p.add_argument("--fix", action="store_true", help="İngilizce kalan kısımları yeniden çevir (LLM_API_KEY gerekir)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    text = args.markdown.read_text(encoding="utf-8")
    fixed, rows = check(text, args.fix, config.LLM_WRITE_MODEL)
    print(f"{'bölüm':<44}{'önce':>6}{'sonra':>7}{'kalan':>8}{'başlık':>8}{'onarılan':>10}{'başarısız':>11}{'token':>7}")
    for r in rows:
        print(
            f"{r['section'][:42]:<44}{r['before']:>6.0%}{r['after']:>7.0%}"
            f"{str(r['left_before']) + '>' + str(r['left_after']):>8}{'EN' if r['heading_english'] else 'TR':>8}"
            f"{r['repaired']:>10}{r['failed']:>11}{r['tokens']:>7}"
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
