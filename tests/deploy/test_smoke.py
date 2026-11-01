from scripts.deploy import smoke

HEALTHY = {"status": "ok", "pipeline_loaded": True, "build": {"commit": "abc"}, "data": {}, "gemini": {}}


def test_smoke_passes_with_a_healthy_site_and_a_working_search(capsys):
    found = {"accepted": True, "results": [{"id": "1"}], "timings_ms": {}}
    assert smoke.run("u", health=lambda url: HEALTHY, find=lambda url: found, sleep=lambda s: None) == 0
    assert "duman testi geçti" in capsys.readouterr().out


def test_smoke_fails_when_search_returns_nothing_or_the_site_never_wakes(capsys):
    empty = {"accepted": True, "results": []}
    assert smoke.run("u", health=lambda url: HEALTHY, find=lambda url: empty, sleep=lambda s: None) == 1
    assert smoke.run("u", wake_timeout=0, health=lambda url: None, find=lambda url: empty, sleep=lambda s: None) == 1
    assert "hazır olmadı" in capsys.readouterr().out


def test_the_same_english_sample_is_drawn_every_week():
    picked = smoke.sample_queries()
    assert {k: len(v) for k, v in picked.items()} == {"title": 30, "synthetic": 30, "offtopic": 30}
    assert all(r["lang"] == "en" for rows in picked.values() for r in rows)
    assert picked == smoke.sample_queries()


QUERIES = smoke.sample_queries(size=10)
LOOKUP = {r["query"]: r for rows in QUERIES.values() for r in rows}


def _finder(miss_titles=0, accept_offtopic=0):
    state = {"title": 0, "offtopic": 0}

    def find(url, query="", k=5):
        row = LOOKUP.get(query)
        if row is None:
            return {"accepted": True, "results": [{"id": "x"}]}
        if row["type"] == "offtopic":
            state["offtopic"] += 1
            return {"accepted": state["offtopic"] <= accept_offtopic, "results": []}
        if row["type"] == "title":
            state["title"] += 1
            if state["title"] <= miss_titles:
                return {"accepted": True, "results": [{"id": "other"}]}
        return {"accepted": True, "results": [{"id": "other"}, {"id": row["target_id"]}]}

    return find


def test_quality_passes_when_known_papers_are_found_and_offtopic_is_rejected(capsys):
    code = smoke.run("u", health=lambda url: HEALTHY, find=_finder(), sleep=lambda s: None, queries=QUERIES)
    out = capsys.readouterr().out
    assert code == 0 and "kalite title/en" in out and "duman testi geçti" in out


def test_quality_fails_below_the_thresholds_and_names_the_misses(capsys):
    assert (
        smoke.run("u", health=lambda url: HEALTHY, find=_finder(miss_titles=2), sleep=lambda s: None, queries=QUERIES)
        == 1
    )
    out = capsys.readouterr().out
    assert "title/en hit@10 0.80 < 0.9" in out and "title-" in out
    assert (
        smoke.run(
            "u", health=lambda url: HEALTHY, find=_finder(accept_offtopic=3), sleep=lambda s: None, queries=QUERIES
        )
        == 1
    )
    assert "offtopic/en kabul oranı 0.30 > 0.2" in capsys.readouterr().out
