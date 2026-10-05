import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from core import config
from core.features import RANK_FEATURES
from core.metrics import rank_of, summarize


def load(features_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cands = pd.read_parquet(features_dir / f"{split}_candidates.parquet")
    queries = pd.read_parquet(features_dir / f"{split}_queries.parquet")
    return cands, queries


def is_valid(paper_id: str, percent: int = 15) -> bool:
    return int(hashlib.md5(f"valid-{paper_id}".encode()).hexdigest(), 16) % 100 < percent


def grouped(cands: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    cands = cands.sort_values("qid", kind="stable")
    sizes = cands.groupby("qid", sort=False).size().tolist()
    return cands[RANK_FEATURES].to_numpy(np.float32), cands["label"].to_numpy(), sizes


def ranks_by(cands: pd.DataFrame, queries: pd.DataFrame, column: str, ascending: bool) -> dict[str, list]:
    out = {}
    for qid, g in cands.groupby("qid", sort=False):
        out[qid] = g.sort_values(column, ascending=ascending, kind="stable")["id"].tolist()
    groups = {}
    for row in queries.itertuples():
        if row.target_id is None or pd.isna(row.target_id):
            continue
        groups.setdefault(f"{row.type}/{row.lang}", []).append(rank_of(row.target_id, out.get(row.qid, [])))
    return groups


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="eval")
    p.add_argument("--model-out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cands, queries = load(args.features_dir, args.train_split)
    _, eval_queries = load(args.features_dir, args.eval_split)
    eval_targets = set(eval_queries.target_id.dropna())
    leaked = set(queries.loc[queries.target_id.isin(eval_targets), "qid"])
    cands = cands[~cands.qid.isin(leaked)]
    if leaked:
        print(f"Hedefi değerlendirme setinde olan {len(leaked)} eğitim sorgusu çıkarıldı.")
    positive = cands.groupby("qid")["label"].max()
    cands = cands[cands.qid.isin(positive[positive > 0].index)]
    target_of = queries.set_index("qid")["target_id"]
    valid_mask = cands.qid.map(lambda q: is_valid(target_of[q]))
    x_tr, y_tr, g_tr = grouped(cands[~valid_mask])
    print(f"Eğitim: {len(g_tr):,} sorgu, doğrulama: {cands[valid_mask].qid.nunique():,} sorgu "
          f"(hedefi aday havuzunda olmayan {int((positive == 0).sum()):,} sorgu çıkarıldı)")

    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.05, num_leaves=31,
                           min_child_samples=20, random_state=args.seed, verbose=-1)
    fit_args = {}
    if valid_mask.any():
        x_va, y_va, g_va = grouped(cands[valid_mask])
        fit_args = {"eval_set": [(x_va, y_va)], "eval_group": [g_va], "eval_at": [10],
                    "callbacks": [lgb.early_stopping(30, verbose=False)]}
    model.fit(x_tr, y_tr, group=g_tr, **fit_args)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.model_out))

    e_cands, e_queries = load(args.features_dir, args.eval_split)
    e_cands = e_cands.copy()
    e_cands["ranker_score"] = model.predict(e_cands[RANK_FEATURES].to_numpy(np.float32))
    dense = ranks_by(e_cands, e_queries, "dense_rank", ascending=True)
    ranker = ranks_by(e_cands, e_queries, "ranker_score", ascending=False)

    report = {
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "train_queries": len(g_tr),
        "feature_importance_gain": dict(zip(RANK_FEATURES, [round(float(v), 1) for v in
                                                            model.booster_.feature_importance("gain")])),
        "dense": {k: summarize(v) for k, v in sorted(dense.items())},
        "ranker": {k: summarize(v) for k, v in sorted(ranker.items())},
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "ranker.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'grup':<14}{'dense@10':>10}{'sıralayıcı@10':>15}{'dense@3':>9}{'sıralayıcı@3':>14}")
    for key in sorted(dense):
        d, r = report["dense"][key], report["ranker"][key]
        print(f"{key:<14}{d['hit@10']:>10.3f}{r['hit@10']:>15.3f}{d['hit@3']:>9.3f}{r['hit@3']:>14.3f}")
    top = sorted(report["feature_importance_gain"].items(), key=lambda kv: -kv[1])[:5]
    print("\nEn etkili özellikler:", ", ".join(f"{k} ({v:.0f})" for k, v in top))
    print(f"Model: {args.model_out}")


if __name__ == "__main__":
    main()
