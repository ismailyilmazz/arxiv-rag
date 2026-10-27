import io
import urllib.error

import pytest

from arxiv_rag.storage import oai


def record(pid, title="A title", abstract="An abstract text.", deleted=False):
    if deleted:
        return f'<record><header status="deleted"><identifier>oai:arXiv.org:{pid}</identifier></header></record>'
    return f"""<record><header><identifier>oai:arXiv.org:{pid}</identifier><datestamp>2026-10-27</datestamp></header>
<metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>{pid}</id><created>2026-10-26</created>
<authors><author><keyname>Doe</keyname><forenames>Jane</forenames></author><author><keyname>Roe</keyname><forenames>R.</forenames><suffix>Jr</suffix></author></authors>
<title>{title}</title><categories>cs.CL cs.LG</categories><license>http://creativecommons.org/licenses/by/4.0/</license>
<abstract>  {abstract}
  continued </abstract></arXiv></metadata></record>"""


def page(records, token=""):
    resumption = f"<resumptionToken>{token}</resumptionToken>" if token else "<resumptionToken/>"
    return f"""<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>{"".join(records)}{resumption}</ListRecords></OAI-PMH>"""


def test_harvest_follows_resumption_tokens_and_skips_deleted_and_invalid_records():
    calls, sleeps = [], []
    replies = iter(
        [
            page([record("2610.00001"), record("2610.00002", deleted=True)], "tok1"),
            page([record("bad-id"), record("2610.00003")]),
        ]
    )

    def get(url, contact):
        calls.append(url)
        return next(replies)

    rows, skipped = oai.harvest("2026-10-20", "2026-10-27", "me@x.org", get, sleeps.append)
    assert [r["id"] for r in rows] == ["2610.00001", "2610.00003"] and skipped == 2
    assert calls[0].startswith(oai.BASE_URL) and "from=2026-10-20" in calls[0] and "metadataPrefix=arXiv" in calls[0]
    assert "resumptionToken=tok1" in calls[1] and sleeps == [oai.DELAY_SECONDS]
    row = rows[0]
    assert row["authors"] == "Jane Doe, R. Roe Jr" and row["primary_category"] == "cs.CL"
    assert row["abstract"] == "An abstract text. continued" and row["published"] == row["updated"] == "2026-10-26"


def test_no_records_is_empty_and_other_errors_raise():
    empty = '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="noRecordsMatch">none</error></OAI-PMH>'
    assert oai.harvest("a", "b", "c", lambda u, c: empty, lambda s: None) == ([], 0)
    bad = '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="badArgument">wrong</error></OAI-PMH>'
    with pytest.raises(RuntimeError, match="badArgument"):
        oai.harvest("a", "b", "c", lambda u, c: bad, lambda s: None)


def test_flow_control_503_waits_for_retry_after(monkeypatch):
    sleeps, answers = [], [urllib.error.HTTPError("u", 503, "busy", {"Retry-After": "7"}, io.BytesIO(b"")), b"<ok/>"]

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(request, timeout=0):
        assert "mailto:me@x.org" in request.get_header("User-agent")
        item = answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return Reply(item)

    monkeypatch.setattr(oai.urllib.request, "urlopen", urlopen)
    assert oai.http_get("https://x", "me@x.org", sleep=sleeps.append) == "<ok/>" and sleeps == [7]


def test_quality_check_flags_too_many_or_empty_records():
    good = [{"id": str(i), "title": "t", "abstract": "a"} for i in range(50)]
    assert oai.check(good) == []
    assert oai.check(good, limit=10)
    assert oai.check(good + [{"id": f"e{i}", "title": "", "abstract": "a"} for i in range(20)])
