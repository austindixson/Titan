from __future__ import annotations


class PermissionError(Exception):
    pass


class PermissionPolicy:
    def __init__(self, mode: str = "allow"):
        self.mode = mode

    def authorize(self, tool_name: str) -> None:
        if self.mode == "allow":
            return
        if tool_name in {"read_file", "todo_get", "todo_set", "memory_get", "memory_add", "memory_remove", "session_recent", "session_search"}:
            return
        raise PermissionError(f"tool '{tool_name}' denied by permission mode '{self.mode}'")
