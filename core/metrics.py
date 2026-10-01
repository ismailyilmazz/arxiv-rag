from typing import Optional, Sequence


def rank_of(target_id: str, result_ids: Sequence[str]) -> Optional[int]:
    for i, pid in enumerate(result_ids, start=1):
        if pid == target_id:
            return i
    return None


def summarize(ranks: Sequence[Optional[int]], k: int = 10) -> dict:
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    hit1 = sum(1 for r in ranks if r == 1) / n
    hitk = sum(1 for r in ranks if r is not None and r <= k) / n
    mrr = sum(1 / r for r in ranks if r is not None and r <= k) / n
    return {"n": n, "hit@1": round(hit1, 4), f"hit@{k}": round(hitk, 4), f"mrr@{k}": round(mrr, 4)}
