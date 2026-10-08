import argparse
import json
from datetime import datetime
from pathlib import Path

from core import config, coverage
from core.db import connect
from core.pipeline import SearchPipeline
from core import writer
from core.survey import build_broad


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--topic", required=True)
    p.add_argument("--type", choices=sorted(writer.BROAD), default="survey")
    p.add_argument("--seeds", nargs="*", default=[])
    p.add_argument("--lang", choices=["en", "tr"], default="en")
    p.add_argument("--n-sources", type=int, default=30)
    p.add_argument("--max-deep", type=int, default=6)
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--gold-survey", default=None)
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    p.add_argument("--cache-db", type=Path, default=config.CACHE_DB_PATH)
    p.add_argument("--vectors-dir", type=Path, default=config.DATA_DIR / "vectors_dev")
    p.add_argument("--models-dir", type=Path, default=config.DATA_DIR / "models")
    p.add_argument("--out-dir", type=Path, default=config.DATA_DIR / "generations")
    args = p.parse_args()

    conn = connect(args.db)
    pipeline = SearchPipeline.load(conn, args.vectors_dir, args.models_dir)
    exclude = list(args.exclude) + ([args.gold_survey] if args.gold_survey else [])
    result = build_broad(conn, connect(args.cache_db), pipeline, args.topic, args.type, args.seeds, lang=args.lang,
                          n_sources=args.n_sources, max_deep=args.max_deep, exclude=tuple(exclude))
    if args.gold_survey:
        gold = coverage.in_corpus(conn, coverage.fetch_gold(args.gold_survey))
        title = conn.execute("SELECT title FROM papers WHERE id = ?", (args.gold_survey,)).fetchone()
        result["coverage"] = {"gold_survey": args.gold_survey, "gold_title": title[0] if title else None,
                              **coverage.coverage(gold, result["candidate_ids"], result["screened_ids"],
                                                  result["cited"])}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.type}-broad_{args.lang}"
    (args.out_dir / f"{stem}.md").write_text(result["markdown"], encoding="utf-8")
    report = {k: v for k, v in result.items() if k != "markdown"}
    (args.out_dir / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Konu: {args.topic} | tür: {args.type} | dil: {args.lang}")
    print(f"Sorgular: {result['queries']}" + (f"  (bekçinin reddettiği: {result['rejected_queries']})"
                                               if result["rejected_queries"] else ""))
    print(f"Aday: {result['candidates']}, elenen sonrası: {result['screened']}, atıflanan: {len(result['cited'])}, "
          f"ileri okuma: {len(result['further_reading'])}")
    for t in result["themes"]:
        print(f"  Tema: {t['name']}  ({t['size']} makale, derin: {', '.join(t['deep']) or '-'})")
    c = result["citations"]
    print(f"Atıf: geçersiz {c['invalid_numbers'] or 'yok'}, atıfsız paragraf {c['uncited_paragraphs']}/{c['paragraphs']}, "
          f"isim-atıf uyuşmazlığı {c['name_mismatches']['count']}, iç dil {c['meta_language']['count']}")
    if "coverage" in result:
        cov = result["coverage"]
        print(f"Kapsama ({cov['gold_survey']}: {cov['gold_title']}): korpustaki hakem kaynağı {cov['gold_in_corpus']}, "
              f"aday {cov['candidates']}, elenen {cov['screened']}, atıflanan {cov['cited']}")
    print(f"Uzunluk: {result['length']['words']} kelime, kesilen bölüm: "
          f"{[r['section'] for r in result['sections'] if r['truncated']] or 'yok'}")
    print(f"Token: {result['tokens']}  |  süre (ms): {result['timings_ms']}")
    print(f"Çıktı: {args.out_dir / (stem + '.md')}")


if __name__ == "__main__":
    main()
