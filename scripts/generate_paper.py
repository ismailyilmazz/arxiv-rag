import argparse
import json
from datetime import datetime
from pathlib import Path

from core import config, writer
from core.db import connect
from core.generate import generate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--type", choices=sorted(writer.DOC_TYPES), default="survey")
    p.add_argument("--lang", choices=sorted(writer.LANGUAGES), default="en")
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    p.add_argument("--cache-db", type=Path, default=config.CACHE_DB_PATH)
    p.add_argument("--out-dir", type=Path, default=config.DATA_DIR / "generations")
    args = p.parse_args()

    result = generate(connect(args.db), connect(args.cache_db), args.ids, doc_type=args.type, lang=args.lang)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.type}_{args.lang}"
    (args.out_dir / f"{stem}.md").write_text(result["markdown"], encoding="utf-8")
    report = {k: v for k, v in result.items() if k != "markdown"}
    (args.out_dir / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Tür: {args.type} | dil: {args.lang} | modeller: {result['models']}")
    for s in result["sources"]:
        print(f"  [{s['n']}] {s['id']}  metin: {s['text_source']}{'  (önbellekten)' if s['cached'] else ''}  {s['title'][:70]}")
    c = result["citations"]
    print(f"Atıf: {c['citations_found']} bulundu, geçersiz numaralar: {c['invalid_numbers'] or 'yok'}, "
          f"atıf almayan kaynak: {c['uncited_sources'] or 'yok'}, "
          f"atıfsız paragraf: {c['uncited_paragraphs']}/{c['paragraphs']}")
    length = result["length"]
    print(f"Uzunluk: {length['words']} kelime (kaynakça hariç), cevap payı {length['output_budget_tokens']} token"
          f"{'  UYARI: metin token sınırında kesildi' if length['truncated'] else ''}")
    print(f"Token: {result['tokens']}  |  süre (ms): {result['timings_ms']}")
    print(f"Çıktı: {args.out_dir / (stem + '.md')}")


if __name__ == "__main__":
    main()
