import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core import config
from core.features import QUERY_FEATURES, RANK_FEATURES

GUARD_FEATURES = [*QUERY_FEATURES, "ranker_top"]


def with_ranker_top(features_dir: Path, split: str, booster: lgb.Booster) -> pd.DataFrame:
    cands = pd.read_parquet(features_dir / f"{split}_candidates.parquet")
    queries = pd.read_parquet(features_dir / f"{split}_queries.parquet")
    cands["score"] = booster.predict(cands[RANK_FEATURES].to_numpy(np.float32))
    queries["ranker_top"] = queries.qid.map(cands.groupby("qid")["score"].max()).fillna(cands["score"].min())
    return queries


def rates(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    accepted = scores >= threshold
    return {
        "false_reject_rate": round(float((~accepted[labels == 1]).mean()), 4),
        "offtopic_reject_rate": round(float((~accepted[labels == 0]).mean()), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--ranker-model", type=Path, required=True)
    p.add_argument("--model-out", type=Path, required=True)
    p.add_argument("--pos-split", default="train")
    p.add_argument("--neg-split", default="offtopic")
    p.add_argument("--eval-split", default="eval")
    p.add_argument("--target-frr", type=float, default=0.05)
    args = p.parse_args()

    booster = lgb.Booster(model_file=str(args.ranker_model))
    pos = with_ranker_top(args.features_dir, args.pos_split, booster)
    neg = with_ranker_top(args.features_dir, args.neg_split, booster)
    train = pd.concat([pos.assign(y=1), neg.assign(y=0)], ignore_index=True)
    evals = with_ranker_top(args.features_dir, args.eval_split, booster)
    evals["y"] = evals.target_id.notna().astype(int)
    print(f"Eğitim: {int(train.y.sum()):,} gerçek sorgu, {int((train.y == 0).sum()):,} konu dışı soru")

    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    model.fit(train[GUARD_FEATURES].to_numpy(np.float32), train.y)
    train_prob = model.predict_proba(train[GUARD_FEATURES].to_numpy(np.float32))[:, 1]
    threshold = float(np.quantile(train_prob[train.y == 1], args.target_frr))
    eval_prob = model.predict_proba(evals[GUARD_FEATURES].to_numpy(np.float32))[:, 1]

    base_threshold = float(np.quantile(train.loc[train.y == 1, "top1_dense"], args.target_frr))
    y = evals.y.to_numpy()
    report = {
        "target_false_reject_rate": args.target_frr,
        "guard": {"roc_auc": round(roc_auc_score(y, eval_prob), 4), "threshold": round(threshold, 4),
                  **rates(eval_prob, y, threshold)},
        "top1_dense_threshold": {"roc_auc": round(roc_auc_score(y, evals.top1_dense), 4),
                                 "threshold": round(base_threshold, 4),
                                 **rates(evals.top1_dense.to_numpy(), y, base_threshold)},
        "coefficients": dict(zip(GUARD_FEATURES, [round(float(c), 3) for c in model[-1].coef_[0]])),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "features": GUARD_FEATURES}, args.model_out)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "guard.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'yöntem':<22}{'ROC-AUC':>9}{'yanlış ret':>12}{'konu dışı ret':>15}")
    for name in ("guard", "top1_dense_threshold"):
        r = report[name]
        print(f"{name:<22}{r['roc_auc']:>9.3f}{r['false_reject_rate']:>12.3f}{r['offtopic_reject_rate']:>15.3f}")
    print(f"\nModel: {args.model_out}")


if __name__ == "__main__":
    main()
