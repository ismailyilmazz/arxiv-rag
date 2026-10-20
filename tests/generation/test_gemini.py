import io
import json
import urllib.error

import pytest

from arxiv_rag import config
from arxiv_rag.generation import gemini
from scripts.generation import check_gemini


class Reply:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.data).encode("utf-8")


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "AQ.test")
    monkeypatch.setattr(config, "GEMINI_AUTH", "bearer")
    monkeypatch.setattr(config, "GEMINI_MODEL", "gemini-test-flash")
    monkeypatch.setattr(config, "GEMINI_RPM", 0)
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    gemini.reset_usage()
    gemini.forget()


def fake_urlopen(monkeypatch, *replies):
    seen, queue = [], list(replies)

    def urlopen(request, timeout=0):
        seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return Reply(item)

    monkeypatch.setattr(gemini.urllib.request, "urlopen", urlopen)
    return seen


def http_error(code):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b'{"error": "quota"}'))


ANSWER = {"choices": [{"message": {"content": "Merhaba"}, "finish_reason": "stop"}], "usage": {"total_tokens": 42}}


def test_missing_key_makes_the_layer_unavailable(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    assert not gemini.available()
    with pytest.raises(gemini.GeminiUnavailable):
        gemini.complete("x")


def test_both_auth_modes_send_the_key_in_the_right_header():
    assert gemini.headers("bearer")["Authorization"] == "Bearer AQ.test"
    header = gemini.headers("header")
    assert header["x-goog-api-key"] == "AQ.test" and "Authorization" not in header


def test_complete_sends_the_model_and_counts_usage(monkeypatch):
    seen = fake_urlopen(monkeypatch, ANSWER)
    assert gemini.complete("Selam", max_tokens=99, model="gemini-test-flash", auth="bearer") == ("Merhaba", 42, "stop")
    body = json.loads(seen[0].data)
    assert seen[0].full_url.endswith("/chat/completions") and body["model"] == "gemini-test-flash"
    assert body["max_tokens"] == 99 and gemini.usage() == {"calls": 1, "tokens": 42, "errors": 0}


def test_chat_reads_json_inside_code_fences(monkeypatch):
    fenced = {"choices": [{"message": {"content": '```json\n{"terms": []}\n```'}}], "usage": {"total_tokens": 5}}
    fake_urlopen(monkeypatch, fenced)
    assert gemini.chat("x", model="m", auth="bearer") == ({"terms": []}, 5)


def test_rate_limit_errors_are_retried_then_reported(monkeypatch):
    fake_urlopen(monkeypatch, http_error(429), http_error(429), http_error(429))
    with pytest.raises(gemini.GeminiUnavailable, match="HTTP 429"):
        gemini.complete("x", model="m", auth="bearer")
    assert gemini.usage()["errors"] == 1


def test_a_retry_that_succeeds_returns_normally(monkeypatch):
    fake_urlopen(monkeypatch, http_error(503), ANSWER)
    assert gemini.complete("x", model="m", auth="bearer")[0] == "Merhaba"


def test_auth_errors_are_not_retried(monkeypatch):
    seen = fake_urlopen(monkeypatch, http_error(401))
    with pytest.raises(gemini.GeminiUnavailable, match="HTTP 401"):
        gemini.complete("x", model="m", auth="bearer")
    assert len(seen) == 1


def test_model_list_drops_the_models_prefix(monkeypatch):
    fake_urlopen(monkeypatch, {"data": [{"id": "models/gemini-2.5-flash"}, {"id": "gemini-3-pro"}]})
    assert gemini.list_models("header") == ["gemini-2.5-flash", "gemini-3-pro"]


def test_requests_per_minute_are_limited(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(config, "GEMINI_RPM", 2)
    monkeypatch.setattr(gemini.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gemini.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    gemini._CALLS.clear()
    for _ in range(3):
        gemini._wait_turn()
    assert clock[0] >= 160


def test_model_pick_prefers_the_configured_then_the_newest_plain_flash():
    models = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3-pro"]
    assert gemini.pick_model(models, "gemini-3-pro") == "gemini-3-pro"
    assert gemini.pick_model(models, "gemini-flash-latest") == "gemini-3.5-flash"
    assert gemini.pick_model(["gemini-pro"], "x") == ""


def test_check_script_reports_the_working_auth(monkeypatch, capsys):
    def list_models(auth):
        if auth == "bearer":
            raise gemini.GeminiUnavailable("HTTP 401: ACCESS_TOKEN_TYPE_UNSUPPORTED")
        return ["gemini-3.5-flash"]

    monkeypatch.setattr(gemini, "list_models", list_models)
    monkeypatch.setattr(gemini, "complete", lambda prompt, max_tokens, model, auth: ("OK", 7, "stop"))
    assert check_gemini.main() == 0
    out = capsys.readouterr().out
    assert "GEMINI_AUTH=header" in out and "GEMINI_MODEL=gemini-3.5-flash" in out and "AQ.test" not in out


MODELS = {"data": [{"id": "models/gemini-3.5-flash"}, {"id": "models/gemini-3.5-flash-lite"}]}


def test_status_without_a_key_does_not_touch_the_network(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    seen = fake_urlopen(monkeypatch)
    assert gemini.status() == {"enabled": False} and seen == []


def test_status_finds_the_working_auth_and_model_once(monkeypatch):
    seen = fake_urlopen(monkeypatch, http_error(401), MODELS, ANSWER)
    info = gemini.status()
    assert info["ok"] and info["auth"] == "header" and info["model"] == "gemini-3.5-flash"
    assert info["errors"] == {"bearer": info["errors"]["bearer"]} and "401" in info["errors"]["bearer"]
    assert gemini.status() == info and len(seen) == 3
    assert "AQ.test" not in str(info)


def test_calls_without_model_use_the_probed_route(monkeypatch):
    seen = fake_urlopen(monkeypatch, MODELS, ANSWER, ANSWER)
    gemini.complete("Selam")
    body = json.loads(seen[2].data)
    assert body["model"] == "gemini-3.5-flash" and seen[2].headers["Authorization"] == "Bearer AQ.test"


def test_failed_probe_is_retried_after_a_while(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(gemini.time, "monotonic", lambda: clock[0])
    fake_urlopen(monkeypatch, http_error(401), http_error(401), MODELS, ANSWER)
    assert gemini.status()["ok"] is False
    assert gemini.status()["ok"] is False
    clock[0] += gemini.RECHECK_SECONDS + 1
    assert gemini.status()["ok"] is True
