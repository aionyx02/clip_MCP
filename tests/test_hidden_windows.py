"""No program the server or the editor starts opens a console window on Windows."""

import ast
from pathlib import Path

import app

# Calls that may leave the flag out: they only run on macOS, or start a program that is a
# window of its own, where a hidden window would hide the program itself.
ALLOWED = {
    ("engine/machine.py", "sysctl"),
    ("engine/resources.py", "vm_stat"),
    ("ui/app.py", "explorer"),
    ("ui/launch.py", "browser"),
}


def _starts(call: ast.Call) -> str:
    """The program a call starts, as far as reading it says."""
    first = call.args[0] if call.args else None
    if isinstance(first, ast.List) and first.elts:
        head = first.elts[0]
        if isinstance(head, ast.Constant):
            return str(head.value)
        return ast.unparse(head)
    return ast.unparse(first) if first is not None else ""


def test_every_program_started_hides_its_window() -> None:
    root = Path(app.__file__).parent
    found = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"
                    and node.func.attr in {"run", "Popen", "check_output", "check_call", "call"}):
                continue
            # `**options` carries the flags where they are worked out elsewhere.
            if any(keyword.arg in (None, "creationflags") for keyword in node.keywords):
                continue
            where = path.relative_to(root).as_posix()
            if not any(where == allowed and name in _starts(node) for allowed, name in ALLOWED):
                found.append(f"{where}:{node.lineno} starts {_starts(node)}")
    assert found == []
