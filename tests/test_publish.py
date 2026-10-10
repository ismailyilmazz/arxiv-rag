import subprocess
import sys

from scripts.publish_results import publish, run


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _setup(tmp_path):
    remote = tmp_path / "remote.git"
    git("init", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    work = tmp_path / "work"
    git("clone", str(remote), str(work), cwd=tmp_path)
    git("config", "user.email", "a@b.c", cwd=work)
    git("config", "user.name", "a", cwd=work)
    (work / "README.md").write_text("x")
    git("add", ".", cwd=work)
    git("commit", "-m", "init", cwd=work)
    git("push", "origin", "HEAD:main", cwd=work)
    return remote, work


def test_publish_copies_commits_and_pushes_even_if_remote_moved(tmp_path):
    remote, work = _setup(tmp_path)
    other = tmp_path / "other"
    git("clone", str(remote), str(other), cwd=tmp_path)
    git("config", "user.email", "a@b.c", cwd=other)
    git("config", "user.name", "a", cwd=other)
    (other / "local.txt").write_text("y")
    git("add", ".", cwd=other)
    git("commit", "-m", "local change", cwd=other)
    git("push", "origin", "HEAD:main", cwd=other)

    source = tmp_path / "generations"
    source.mkdir()
    (source / "a_survey-broad_en.md").write_text("# T")
    (source / "a_survey-broad_en.json").write_text("{}")
    (source / "notes.txt").write_text("skip")
    message = publish(work, source, str(remote))
    assert message.startswith("2 dosya repoya gönderildi (main)")
    files = subprocess.run(["git", "--git-dir", str(remote), "ls-tree", "-r", "main", "--name-only"],
                           capture_output=True, text=True).stdout.split()
    assert {"eval/generations/a_survey-broad_en.md", "eval/generations/a_survey-broad_en.json",
            "local.txt"} <= set(files) and "eval/generations/notes.txt" not in files


def test_publish_without_files_and_secret_is_masked(tmp_path):
    remote, work = _setup(tmp_path)
    (tmp_path / "empty").mkdir()
    assert publish(work, tmp_path / "empty", str(remote)) == "Gönderilecek dosya yok."
    code, out = run([sys.executable, "-c", "print('push to https://x-access-token:SECRET123@github.com')"],
                    tmp_path, "SECRET123")
    assert code == 0 and "SECRET123" not in out and "***" in out