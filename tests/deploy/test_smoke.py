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
