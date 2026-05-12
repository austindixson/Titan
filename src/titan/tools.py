from __future__ import annotations
import json
import subprocess
import os
import re
import sys
import time
from urllib import parse, request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from .types import ToolResult


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable[[dict], str]] = {}
        self.cwd: Path = Path.cwd()

    def register(self, name: str, fn: Callable[[dict], str]) -> None:
        self._tools[name] = fn

    def definitions(self) -> list[dict]:
        specs = {
            "read_file": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to text file"},
                    "offset": {"type": "integer", "description": "1-indexed start line", "default": 1},
                    "limit": {"type": "integer", "description": "Max lines to read", "default": 500},
                },
                "required": ["path"],
            },
            "write_file": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file"},
                    "content": {"type": "string", "description": "Full file content to write"},
                },
                "required": ["path", "content"],
            },
            "shell": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 60},
                },
                "required": ["command"],
            },
            "cd": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to switch into"},
                },
                "required": ["path"],
            },
            "todo_get": {
                "type": "object",
                "properties": {},
            },
            "todo_set": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "content": {"type": "string"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                            },
                            "required": ["id", "content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
            "memory_get": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional keyword filter"},
                },
            },
            "memory_add": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Durable memory note"},
                },
                "required": ["content"],
            },
            "memory_remove": {
                "type": "object",
                "properties": {
                    "contains": {"type": "string", "description": "Remove entries containing this substring"},
                },
                "required": ["contains"],
            },
            "session_recent": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max traces to return", "default": 5},
                },
            },
            "session_search": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Case-insensitive substring query"},
                    "limit": {"type": "integer", "description": "Max traces to return", "default": 5},
                },
                "required": ["query"],
            },
            "web_search": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5},
                },
                "required": ["query"],
            },
            "browser_navigate": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "HTTP/HTTPS URL"},
                    "timeout": {"type": "integer", "description": "Timeout seconds", "default": 20},
                },
                "required": ["url"],
            },
            "delegate_task": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Task goal"},
                    "context": {"type": "string", "description": "Optional context for the worker"},
                    "command": {"type": "string", "description": "Optional shell command to run as the local worker"},
                    "timeout": {"type": "integer", "description": "Timeout seconds", "default": 300},
                },
                "required": ["goal"],
            },
            "cronjob": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "One of create|list|pause|resume|remove|run"},
                    "job_id": {"type": "string", "description": "Existing job id for pause|resume|remove|run"},
                    "name": {"type": "string", "description": "Human-friendly job name"},
                    "schedule": {"type": "string", "description": "Schedule label or cron expression"},
                    "command": {"type": "string", "description": "Local shell command to run for this job"},
                    "timeout": {"type": "integer", "description": "Timeout seconds", "default": 300},
                },
                "required": ["action"],
            },
        }
        out = []
        for k in self._tools:
            out.append({
                "type": "function",
                "function": {
                    "name": k,
                    "description": k,
                    "parameters": specs.get(k, {"type": "object", "properties": {}}),
                },
            })
        return out

    def execute(self, call_id: str, name: str, arguments: dict) -> ToolResult:
        if name not in self._tools:
            return ToolResult(call_id=call_id, tool_name=name, content=f"unknown tool: {name}", is_error=True)
        try:
            out = self._tools[name](arguments)
            return ToolResult(call_id=call_id, tool_name=name, content=out, is_error=False)
        except Exception as e:
            return ToolResult(call_id=call_id, tool_name=name, content=str(e), is_error=True)


def read_file_tool(args: dict) -> str:
    path = Path(args["path"]).expanduser()
    offset = int(args.get("offset", 1))
    limit = int(args.get("limit", 500))
    lines = path.read_text().splitlines()
    chunk = lines[offset - 1 : offset - 1 + limit]
    return "\n".join(f"{i+offset}|{line}" for i, line in enumerate(chunk))


def write_file_tool(args: dict) -> str:
    path = Path(args["path"]).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"])
    return f"wrote {path}"


def shell_tool(args: dict) -> str:
    raise RuntimeError("shell tool must be bound through registry for cwd-aware execution")


def make_shell_tool(reg: ToolRegistry):
    def _run(args: dict) -> str:
        cmd = args["command"]
        timeout = int(args.get("timeout", 120))
        stripped = cmd.strip()

        if stripped == "pwd":
            return json.dumps({"exit_code": 0, "cwd": str(reg.cwd), "stdout": str(reg.cwd), "stderr": ""})

        if stripped.startswith("cd ") and "&&" not in stripped and ";" not in stripped:
            target_raw = stripped[3:].strip().strip('"').strip("'")
            if not target_raw:
                target = Path.home()
            else:
                target = Path(target_raw).expanduser()
                if not target.is_absolute():
                    target = (reg.cwd / target).resolve()
            if not target.exists() or not target.is_dir():
                return json.dumps({"exit_code": 1, "cwd": str(reg.cwd), "stdout": "", "stderr": f"not a directory: {target}"})
            reg.cwd = target.resolve()
            return json.dumps({"exit_code": 0, "cwd": str(reg.cwd), "stdout": str(reg.cwd), "stderr": ""})

        p = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=timeout, cwd=str(reg.cwd))
        data = {"exit_code": p.returncode, "cwd": str(reg.cwd), "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
        return json.dumps(data)

    return _run


def make_cd_tool(reg: ToolRegistry):
    def _cd(args: dict) -> str:
        path = args["path"]
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = (reg.cwd / target).resolve()
        if not target.exists() or not target.is_dir():
            raise RuntimeError(f"not a directory: {target}")
        reg.cwd = target.resolve()
        return f"cwd changed to {reg.cwd}"

    return _cd


def _todo_path(reg: ToolRegistry) -> Path:
    p = (reg.cwd / ".titan" / "todo.json").resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _memory_path(reg: ToolRegistry) -> Path:
    p = (reg.cwd / ".titan" / "memory.json").resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def make_todo_get_tool(reg: ToolRegistry):
    def _todo_get(args: dict) -> str:
        p = _todo_path(reg)
        if not p.exists():
            return json.dumps({"todos": []})
        try:
            data = json.loads(p.read_text())
            todos = data.get("todos", []) if isinstance(data, dict) else []
            if not isinstance(todos, list):
                todos = []
            return json.dumps({"todos": todos})
        except Exception:
            return json.dumps({"todos": []})

    return _todo_get


def make_todo_set_tool(reg: ToolRegistry):
    def _todo_set(args: dict) -> str:
        raw = args.get("todos", [])
        if not isinstance(raw, list):
            raise RuntimeError("todos must be a list")

        clean: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            _id = str(item.get("id", "")).strip()
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).strip()
            if not _id or not content:
                continue
            if status not in {"pending", "in_progress", "completed", "cancelled"}:
                status = "pending"
            clean.append({"id": _id, "content": content, "status": status})

        p = _todo_path(reg)
        p.write_text(json.dumps({"todos": clean}, indent=2) + "\n")
        return json.dumps({"saved": len(clean)})

    return _todo_set


def _read_memory_entries(reg: ToolRegistry) -> list[str]:
    p = _memory_path(reg)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        vals = data.get("entries", []) if isinstance(data, dict) else []
        return [str(v) for v in vals if str(v).strip()]
    except Exception:
        return []


def _write_memory_entries(reg: ToolRegistry, entries: list[str]) -> None:
    p = _memory_path(reg)
    p.write_text(json.dumps({"entries": entries}, indent=2) + "\n")


def make_memory_get_tool(reg: ToolRegistry):
    def _memory_get(args: dict) -> str:
        entries = _read_memory_entries(reg)
        q = str(args.get("query", "")).strip().lower()
        if q:
            entries = [e for e in entries if q in e.lower()]
        return json.dumps({"entries": entries})

    return _memory_get


def make_memory_add_tool(reg: ToolRegistry):
    def _memory_add(args: dict) -> str:
        content = str(args.get("content", "")).strip()
        if not content:
            raise RuntimeError("content is required")
        entries = _read_memory_entries(reg)
        entries.append(content)
        # keep bounded
        entries = entries[-200:]
        _write_memory_entries(reg, entries)
        return json.dumps({"saved": True, "count": len(entries)})

    return _memory_add


def make_memory_remove_tool(reg: ToolRegistry):
    def _memory_remove(args: dict) -> str:
        needle = str(args.get("contains", "")).strip().lower()
        if not needle:
            raise RuntimeError("contains is required")
        entries = _read_memory_entries(reg)
        kept = [e for e in entries if needle not in e.lower()]
        removed = len(entries) - len(kept)
        _write_memory_entries(reg, kept)
        return json.dumps({"removed": removed, "count": len(kept)})

    return _memory_remove


def _session_rows(reg: ToolRegistry) -> list[dict[str, Any]]:
    p = (reg.cwd / ".titan" / "session.jsonl").resolve()
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def make_session_recent_tool(reg: ToolRegistry):
    def _session_recent(args: dict) -> str:
        limit = max(1, int(args.get("limit", 5)))
        rows = _session_rows(reg)
        by_trace: dict[str, dict[str, Any]] = {}
        for r in rows:
            tid = str(r.get("trace_id", "")).strip()
            if not tid:
                continue
            ts = int(r.get("ts", 0) or 0)
            role = str(r.get("role", ""))
            content = str(r.get("content", "")).strip()
            cur = by_trace.setdefault(tid, {"trace_id": tid, "ts": ts, "preview": ""})
            if ts >= int(cur.get("ts", 0) or 0):
                cur["ts"] = ts
            if role == "user" and content and not cur["preview"]:
                cur["preview"] = content[:120]
        items = sorted(by_trace.values(), key=lambda x: int(x.get("ts", 0) or 0), reverse=True)[:limit]
        return json.dumps({"items": items})

    return _session_recent


def make_session_search_tool(reg: ToolRegistry):
    def _session_search(args: dict) -> str:
        query = str(args.get("query", "")).strip().lower()
        if not query:
            raise RuntimeError("query is required")
        limit = max(1, int(args.get("limit", 5)))
        rows = _session_rows(reg)
        score: dict[str, int] = {}
        preview: dict[str, str] = {}
        ts_by_trace: dict[str, int] = {}
        for r in rows:
            tid = str(r.get("trace_id", "")).strip()
            if not tid:
                continue
            content = str(r.get("content", ""))
            role = str(r.get("role", ""))
            ts = int(r.get("ts", 0) or 0)
            ts_by_trace[tid] = max(ts_by_trace.get(tid, 0), ts)
            if query in content.lower():
                score[tid] = score.get(tid, 0) + 1
                if tid not in preview and role == "user":
                    preview[tid] = content.strip()[:120]
        ranked = sorted(score.items(), key=lambda kv: (kv[1], ts_by_trace.get(kv[0], 0)), reverse=True)[:limit]
        items = [{"trace_id": tid, "score": n, "ts": ts_by_trace.get(tid, 0), "preview": preview.get(tid, "")} for tid, n in ranked]
        return json.dumps({"items": items})

    return _session_search


def web_search_tool(args: dict) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        raise RuntimeError("query is required")
    limit = max(1, int(args.get("limit", 5)))
    url = "https://duckduckgo.com/html/?q=" + parse.quote_plus(query)
    req = request.Request(url, headers={"User-Agent": "Titan/1.0"})
    try:
        with request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode(errors="ignore")
    except Exception as e:
        return json.dumps({"query": query, "results": [], "error": str(e)})

    matches = re.findall(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.IGNORECASE | re.DOTALL)
    results: list[dict[str, str]] = []
    for href, title_html in matches:
        title = re.sub(r"<[^>]+>", "", title_html)
        title = re.sub(r"\s+", " ", title).strip()
        results.append({"title": title, "url": href})
        if len(results) >= limit:
            break
    return json.dumps({"query": query, "results": results})


def browser_navigate_tool(args: dict) -> str:
    url = str(args.get("url", "")).strip()
    if not url:
        raise RuntimeError("url is required")
    timeout = int(args.get("timeout", 20))
    req = request.Request(url, headers={"User-Agent": "Titan/1.0"})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode(errors="ignore")
        final_url = resp.geturl()
        status = int(getattr(resp, "status", 200) or 200)
    title_match = re.search(r"<title[^>]*>(.*?)</title>", body, flags=re.IGNORECASE | re.DOTALL)
    title = re.sub(r"\s+", " ", (title_match.group(1) if title_match else "")).strip()
    preview = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()[:400]
    return json.dumps({"url": final_url, "status": status, "title": title, "content_preview": preview})


def make_delegate_task_tool(reg: ToolRegistry):
    def _delegate(args: dict) -> str:
        goal = str(args.get("goal", "")).strip()
        if not goal:
            raise RuntimeError("goal is required")
        context = str(args.get("context", "")).strip()
        timeout = max(1, int(args.get("timeout", 300)))
        delegate_dir = (reg.cwd / ".titan" / "delegates").resolve()
        delegate_dir.mkdir(parents=True, exist_ok=True)
        delegate_id = f"delegate-{int(time.time() * 1000)}-{os.getpid()}"
        requested_command = str(args.get("command", "")).strip()
        env_command = os.getenv("TITAN_DELEGATE_COMMAND", "").strip()
        command = requested_command or env_command

        env = os.environ.copy()
        env["TITAN_DELEGATE_ID"] = delegate_id
        env["TITAN_DELEGATE_GOAL"] = goal
        env["TITAN_DELEGATE_CONTEXT"] = context
        env["TITAN_DELEGATE_CWD"] = str(reg.cwd)

        if command:
            cmd = command
        else:
            cmd = (
                f"{sys.executable!r} -c "
                "'import json, os; "
                "print(json.dumps({\"summary\": \"local delegate recorded task\", "
                "\"goal\": os.environ.get(\"TITAN_DELEGATE_GOAL\", \"\")}))'"
            )

        record = {
            "id": delegate_id,
            "goal": goal,
            "context": context,
            "status": "running",
            "command": cmd,
            "cwd": str(reg.cwd),
            "started_at": int(time.time()),
        }
        record_path = delegate_dir / f"{delegate_id}.json"
        record_path.write_text(json.dumps(record, indent=2) + "\n")

        try:
            proc = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=timeout, cwd=str(reg.cwd), env=env)
            record.update({
                "status": "completed" if proc.returncode == 0 else "failed",
                "exit_code": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "finished_at": int(time.time()),
            })
        except subprocess.TimeoutExpired as e:
            record.update({
                "status": "timeout",
                "exit_code": None,
                "stdout": (e.stdout or "").strip() if isinstance(e.stdout, str) else "",
                "stderr": (e.stderr or "").strip() if isinstance(e.stderr, str) else "",
                "finished_at": int(time.time()),
            })

        record_path.write_text(json.dumps(record, indent=2) + "\n")
        return json.dumps({
            "status": record["status"],
            "tool": "delegate_task",
            "id": delegate_id,
            "goal": goal,
            "record_path": str(record_path),
            "exit_code": record.get("exit_code"),
            "stdout": record.get("stdout", "")[:2000],
            "stderr": record.get("stderr", "")[:2000],
        })

    return _delegate


def make_cronjob_tool(reg: ToolRegistry):
    def jobs_path() -> Path:
        p = (reg.cwd / ".titan" / "cronjobs.json").resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def load_jobs() -> dict[str, Any]:
        p = jobs_path()
        if not p.exists():
            return {"jobs": []}
        try:
            data = json.loads(p.read_text())
            return data if isinstance(data, dict) and isinstance(data.get("jobs"), list) else {"jobs": []}
        except Exception:
            return {"jobs": []}

    def save_jobs(data: dict[str, Any]) -> None:
        jobs_path().write_text(json.dumps(data, indent=2) + "\n")

    def find_job(data: dict[str, Any], job_id: str) -> dict[str, Any] | None:
        for job in data.get("jobs", []):
            if str(job.get("id", "")) == job_id:
                return job
        return None

    def _cronjob(args: dict) -> str:
        action = str(args.get("action", "")).strip().lower()
        if not action:
            raise RuntimeError("action is required")
        data = load_jobs()

        if action == "list":
            return json.dumps({"status": "ok", "jobs": data.get("jobs", [])})

        if action == "create":
            command = str(args.get("command", "")).strip()
            if not command:
                raise RuntimeError("command is required for create")
            job_id = f"job-{int(time.time() * 1000)}-{os.getpid()}"
            job = {
                "id": job_id,
                "name": str(args.get("name", "")).strip() or job_id,
                "schedule": str(args.get("schedule", "")).strip() or "manual",
                "command": command,
                "timeout": max(1, int(args.get("timeout", 300))),
                "paused": False,
                "created_at": int(time.time()),
                "last_run": None,
            }
            data.setdefault("jobs", []).append(job)
            save_jobs(data)
            return json.dumps({"status": "created", "job": job})

        job_id = str(args.get("job_id", "")).strip()
        if not job_id:
            raise RuntimeError("job_id is required")
        job = find_job(data, job_id)
        if job is None:
            raise RuntimeError(f"job not found: {job_id}")

        if action in {"pause", "resume"}:
            job["paused"] = action == "pause"
            save_jobs(data)
            return json.dumps({"status": action + "d", "job": job})

        if action == "remove":
            data["jobs"] = [j for j in data.get("jobs", []) if str(j.get("id", "")) != job_id]
            save_jobs(data)
            return json.dumps({"status": "removed", "job_id": job_id})

        if action == "run":
            if bool(job.get("paused")):
                return json.dumps({"status": "paused", "job_id": job_id})
            proc = subprocess.run(
                str(job.get("command", "")),
                shell=True,
                text=True,
                capture_output=True,
                timeout=int(job.get("timeout", 300) or 300),
                cwd=str(reg.cwd),
            )
            result = {
                "status": "completed" if proc.returncode == 0 else "failed",
                "exit_code": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "finished_at": int(time.time()),
            }
            job["last_run"] = result
            save_jobs(data)
            return json.dumps({"status": result["status"], "job_id": job_id, "result": result})

        raise RuntimeError("unknown cronjob action: " + action)

    return _cronjob


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("read_file", read_file_tool)
    reg.register("write_file", write_file_tool)
    reg.register("shell", make_shell_tool(reg))
    reg.register("cd", make_cd_tool(reg))
    reg.register("todo_get", make_todo_get_tool(reg))
    reg.register("todo_set", make_todo_set_tool(reg))
    reg.register("memory_get", make_memory_get_tool(reg))
    reg.register("memory_add", make_memory_add_tool(reg))
    reg.register("memory_remove", make_memory_remove_tool(reg))
    reg.register("session_recent", make_session_recent_tool(reg))
    reg.register("session_search", make_session_search_tool(reg))
    reg.register("web_search", web_search_tool)
    reg.register("browser_navigate", browser_navigate_tool)
    reg.register("delegate_task", make_delegate_task_tool(reg))
    reg.register("cronjob", make_cronjob_tool(reg))
    return reg
