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

    def hit(cut: int) -> float:
        return round(sum(1 for r in ranks if r is not None and r <= cut) / n, 4)

    mrr = sum(1 / r for r in ranks if r is not None and r <= k) / n
    return {"n": n, "hit@1": hit(1), "hit@3": hit(3), f"hit@{k}": hit(k), f"mrr@{k}": round(mrr, 4)}
