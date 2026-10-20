import argparse
import json
from pathlib import Path

from arxiv_rag import config
from arxiv_rag.generation import language, writer


def retranslate(report: dict, model: str) -> tuple[str, dict]:
    draft = report.get("draft")
    if not draft or not draft.get("sections"):
        raise ValueError("Raporda İngilizce taslak yok; bu rapor taslak saklanmaya başlamadan önce üretilmiş.")
    lang = draft["lang"]
    if lang == "en":
        raise ValueError("Taslak İngilizce bir belgeye ait; çevrilecek bir şey yok.")
    prefix = writer.DOCUMENT_TYPES[report["doc_type"]]["theme_prefix"][lang]
    themes = [{"name": en, "name_local": local} for en, local in (draft.get("names") or {}).items()]
    names, heading_tokens, from_plan = writer.local_names(themes, lang, model)
    sections = []
    for s in draft["sections"]:
        theme = s["key"][6:] if s["key"].startswith("theme:") else None
        text, _ = writer.drop_repeated_heading(s["text"], [s["heading"], theme])
        sections.append({**s, "text": text, "heading_out": prefix + names[theme] if theme else s["heading_out"]})
    texts, rows, tokens, glossary = writer.translate_sections(
        sections, draft["label"], lang, model, names, titles=draft.get("titles") or []
    )
    body = "\n\n".join(f"## {s['heading_out']}\n\n{texts[s['key']]}" for s in sections)
    summary = {
        "tokens": tokens + heading_tokens,
        "headings": {
            "translated": sum(not writer.same_text(local, en) for en, local in names.items()),
            "of": len(names),
            "from_plan": from_plan,
        },
        "glossary": glossary,
        "sections": rows,
        "language": language.audit(body),
        "terms": language.term_audit(body, glossary),
    }
    labels = writer.english_labels(body, draft.get("titles") or [])
    summary["language"].update(english_labels=len(labels), label_examples=[t[:80] for t in labels[:5]])
    title = writer.clean_title(draft.get("title", ""), report.get("topic", ""))
    return f"# {title}\n\n{body}\n", summary


def main() -> None:
    p = argparse.ArgumentParser(
        description="Belge raporundaki İngilizce taslağı yeniden çevirir; yazım aşaması tekrarlanmaz."
    )
    p.add_argument("report", type=Path, help="Belge raporu (JSON), /api/jobs/<id>/report çıktısı")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--model", default=None)
    args = p.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    markdown, summary = retranslate(report, args.model or config.LLM_WRITE_MODEL)
    out = args.out or args.report.with_name(args.report.stem + "-yeniden-ceviri.md")
    out.write_text(markdown, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'bölüm':<52}{'deneme':>8}{'ingilizce kalan':>17}{'token':>8}")
    for r in summary["sections"]:
        print(f"{r['section'][:50]:<52}{r.get('attempts', '-'):>8}{r.get('english_left', '-'):>17}{r['tokens']:>8}")
    print("\nBaşlıklar:", summary["headings"])
    print("Sözlük:", len(summary["glossary"]), "terim")
    print("Dil:", summary["language"])
    print("Terimler:", summary["terms"])
    print("Toplam token:", summary["tokens"])
    print(f"\nÇeviri: {out}\nÖzet: {out.with_suffix('.json')}")
    print("Not: atıf numaraları taslaktaki numaralardır; kaynakça eklenmez.")


if __name__ == "__main__":
    main()
