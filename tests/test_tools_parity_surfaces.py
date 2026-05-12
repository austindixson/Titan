import json
import sys

from titan.tools import default_registry


class _Resp:
    def __init__(self, body: str, url: str = "https://example.com/page", status: int = 200):
        self._body = body.encode()
        self._url = url
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body

    def geturl(self):
        return self._url


def test_parity_surface_tools_registered():
    reg = default_registry()
    names = {d["function"]["name"] for d in reg.definitions()}
    assert "web_search" in names
    assert "browser_navigate" in names
    assert "delegate_task" in names
    assert "cronjob" in names


def test_web_search_and_browser_navigate(monkeypatch):
    reg = default_registry()

    def fake_urlopen(req, timeout=0):
        url = req.full_url
        if "duckduckgo.com" in url:
            return _Resp('<a class="result__a" href="https://a.test">Alpha Result</a>')
        return _Resp("<html><title>Hello World</title><body>Sample body</body></html>")

    monkeypatch.setattr("titan.tools.request.urlopen", fake_urlopen)

    s = reg.execute("p1", "web_search", {"query": "alpha", "limit": 1})
    assert s.is_error is False
    sd = json.loads(s.content)
    assert sd["query"] == "alpha"
    assert len(sd["results"]) == 1
    assert sd["results"][0]["title"] == "Alpha Result"

    n = reg.execute("p2", "browser_navigate", {"url": "https://example.com"})
    assert n.is_error is False
    nd = json.loads(n.content)
    assert nd["status"] == 200
    assert nd["title"] == "Hello World"


def test_delegate_and_cronjob_parity_surface_messages(tmp_path):
    reg = default_registry()
    reg.cwd = tmp_path

    d = reg.execute("p3", "delegate_task", {"goal": "do x"})
    assert d.is_error is False
    dd = json.loads(d.content)
    assert dd["status"] == "completed"
    assert dd["tool"] == "delegate_task"
    assert dd["goal"] == "do x"
    assert "local delegate recorded task" in dd["stdout"]
    record = json.loads((tmp_path / ".titan" / "delegates" / f"{dd['id']}.json").read_text())
    assert record["status"] == "completed"
    assert record["goal"] == "do x"

    c = reg.execute("p4", "cronjob", {"action": "list"})
    assert c.is_error is False
    cd = json.loads(c.content)
    assert cd["status"] == "ok"
    assert cd["jobs"] == []


def test_cronjob_create_run_pause_resume_remove(tmp_path):
    reg = default_registry()
    reg.cwd = tmp_path

    created = reg.execute("c1", "cronjob", {"action": "create", "name": "hello", "schedule": "manual", "command": "printf cron-ok"})
    assert created.is_error is False
    job = json.loads(created.content)["job"]
    assert job["name"] == "hello"

    ran = reg.execute("c2", "cronjob", {"action": "run", "job_id": job["id"]})
    rd = json.loads(ran.content)
    assert rd["status"] == "completed"
    assert rd["result"]["stdout"] == "cron-ok"

    paused = reg.execute("c3", "cronjob", {"action": "pause", "job_id": job["id"]})
    assert json.loads(paused.content)["job"]["paused"] is True
    paused_run = reg.execute("c4", "cronjob", {"action": "run", "job_id": job["id"]})
    assert json.loads(paused_run.content)["status"] == "paused"

    resumed = reg.execute("c5", "cronjob", {"action": "resume", "job_id": job["id"]})
    assert json.loads(resumed.content)["job"]["paused"] is False
    removed = reg.execute("c6", "cronjob", {"action": "remove", "job_id": job["id"]})
    assert json.loads(removed.content)["status"] == "removed"


def test_delegate_task_runs_configured_local_worker(tmp_path):
    reg = default_registry()
    reg.cwd = tmp_path

    d = reg.execute(
        "p5",
        "delegate_task",
        {
            "goal": "inspect module",
            "context": "src/titan/tools.py",
            "command": f"{sys.executable} -c 'import os; print(os.environ[\"TITAN_DELEGATE_GOAL\"] + \" :: \" + os.environ[\"TITAN_DELEGATE_CONTEXT\"])'",
        },
    )

    assert d.is_error is False
    dd = json.loads(d.content)
    assert dd["status"] == "completed"
    assert dd["stdout"] == "inspect module :: src/titan/tools.py"
