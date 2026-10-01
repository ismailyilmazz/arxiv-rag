from core.metrics import rank_of, summarize
from core.splits import split_of


def test_split_is_deterministic_and_about_one_percent():
    ids = [f"2310.{i:05d}" for i in range(20000)]
    assert [split_of(i) for i in ids] == [split_of(i) for i in ids]
    share = sum(split_of(i) == "test" for i in ids) / len(ids)
    assert 0.007 < share < 0.013


def test_rank_of():
    assert rank_of("b", ["a", "b", "c"]) == 2
    assert rank_of("z", ["a", "b"]) is None


def test_summarize():
    # sıralar: 1, 4, bulunamadı, 12 (ilk 10 dışında)
    m = summarize([1, 4, None, 12], k=10)
    assert m == {"n": 4, "hit@1": 0.25, "hit@10": 0.5, "mrr@10": round((1 + 0.25) / 4, 4)}
