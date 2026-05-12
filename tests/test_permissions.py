from titan.permissions import PermissionPolicy, PermissionError


def test_prompt_denies_shell():
    p = PermissionPolicy("prompt")
    p.authorize("read_file")
    p.authorize("todo_get")
    p.authorize("todo_set")
    p.authorize("memory_get")
    p.authorize("memory_add")
    p.authorize("memory_remove")
    p.authorize("session_recent")
    p.authorize("session_search")
    try:
        p.authorize("shell")
        assert False, "shell should be denied"
    except PermissionError:
        pass
