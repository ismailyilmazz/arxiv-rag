import pytest

from arxiv_rag.search.pipeline import SearchPipeline, fuse
from arxiv_rag.search.translate import QueryTranslator

TR = "büyük dil modellerinde halüsinasyon tespiti"


class FakeLLM:
    def __init__(self, reply='"Hallucination detection in large language models"\n'):
        self.reply, self.calls = reply, []

    def __call__(self, prompt, model, max_tokens, reasoning_effort):
        self.calls.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply, 42, "stop"


def test_english_query_is_not_translated():
    llm = FakeLLM()
    assert QueryTranslator(complete=llm)("hallucination detection in large language models") is None
    assert llm.calls == []


def test_turkish_query_is_translated_cleaned_and_cached():
    llm = FakeLLM()
    t = QueryTranslator(complete=llm)
    assert t(TR) == "Hallucination detection in large language models"
    assert t("  büyük dil   modellerinde halüsinasyon tespiti ") == "Hallucination detection in large language models"
    assert len(llm.calls) == 1 and TR in llm.calls[0] and t.tokens == 42


def test_file_cache_survives_a_new_translator(tmp_path):
    cache = tmp_path / "ceviri.json"
    QueryTranslator(complete=FakeLLM(), cache_path=cache)(TR)
    llm = FakeLLM()
    assert QueryTranslator(complete=llm, cache_path=cache)(TR) == "Hallucination detection in large language models"
    assert llm.calls == []


@pytest.mark.parametrize("reply", [RuntimeError("rate limit"), "", "x" * 1000])
def test_bad_translation_falls_back_to_original(reply):
    t = QueryTranslator(complete=FakeLLM(reply))
    assert t(TR) is None
    assert t.failures == (1 if isinstance(reply, Exception) else 0)


def test_fuse_rewards_papers_found_by_both():
    original = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    english = [{"id": "c"}, {"id": "d"}, {"id": "a"}]
    assert [r["id"] for r in fuse([original, english], 4)] == ["a", "c", "b", "d"]


def make_pipeline(translation, translated):
    pipeline = object.__new__(SearchPipeline)
    pipeline.translation, pipeline.translator = translation, (lambda text: translated)
    seen = []

    def fake_search(text, k):
        seen.append((text, k))
        ids = ["en1", "en2", "both"] if text == "english query" else ["tr1", "both", "tr2"]
        return {
            "results": [{"id": i, "score": 1.0, "dense_score": 0.5} for i in ids],
            "accepted": True,
            "guard_prob": 0.9,
            "timings_ms": {"total_ms": 10.0},
        }

    pipeline._search = fake_search
    return pipeline, seen


def test_off_mode_searches_the_original():
    pipeline, seen = make_pipeline("off", "english query")
    out = pipeline.search(TR, k=3)
    assert seen == [(TR, 3)] and out["translated_query"] is None


def test_replace_mode_searches_only_the_translation():
    pipeline, seen = make_pipeline("replace", "english query")
    out = pipeline.search(TR, k=3)
    assert seen == [("english query", 3)] and out["translated_query"] == "english query"


def test_fuse_mode_searches_both_and_merges():
    pipeline, seen = make_pipeline("fuse", "english query")
    out = pipeline.search(TR, k=2)
    assert seen == [(TR, 6), ("english query", 6)]
    assert out["results"][0]["id"] == "both" and out["timings_ms"]["total_ms"] == 20.0


def test_untranslated_query_falls_back_to_original_in_any_mode():
    pipeline, seen = make_pipeline("fuse", None)
    pipeline.search("mixture of experts", k=3)
    assert seen == [("mixture of experts", 3)]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="çeviri modu"):
        SearchPipeline(None, None, None, None, None, None, translator=object(), translation="maybe")
