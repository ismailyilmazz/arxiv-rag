from scripts.deploy import wait_live


def fixed(data):
    return lambda url: data


def test_empty_queue_or_sleeping_site_counts_as_idle():
    assert wait_live.idle("u", fixed({"queue": {"plan": 0, "document": 0}}))[0]
    assert wait_live.idle("u", fixed(None))[0]
    busy, note = wait_live.idle("u", fixed({"queue": {"plan": 0, "document": 1}}))
    assert not busy and "document" in note


def test_new_version_is_live_only_with_the_same_commit_and_a_loaded_pipeline():
    good = {"build": {"commit": "abc"}, "pipeline_loaded": True}
    assert wait_live.live("u", "abc", fixed(good))[0]
    assert not wait_live.live("u", "abc", fixed({**good, "pipeline_loaded": False}))[0]
    assert not wait_live.live("u", "abc", fixed({**good, "build": {"commit": "old"}}))[0]
    assert not wait_live.live("u", "abc", fixed(None))[0]


def test_run_waits_until_the_site_changes(capsys):
    answers = iter(
        [
            {"build": {"commit": "old"}, "pipeline_loaded": True},
            None,
            {"build": {"commit": "abc"}, "pipeline_loaded": True},
        ]
    )
    assert wait_live.run("live", "u", "abc", 600, 1, health=lambda url: next(answers), sleep=lambda s: None) == 0
    out = capsys.readouterr().out
    assert "canlı commit old" in out and "henüz cevap vermiyor" in out and "canlıda abc" in out


def test_run_reports_a_postponed_deploy_when_the_queue_stays_busy(capsys):
    busy = fixed({"queue": {"document": 2}})
    assert wait_live.run("idle", "u", "", 0, 1, health=busy, sleep=lambda s: None) == 1
    assert "yükleme ertelendi" in capsys.readouterr().out
