"""Keep the README's frozen numbers and CI commands aligned with the tree.

Counts live in three lines in README.md. This module recomputes them. CI runs
it because it is named ``test_*.py``. It does not rewrite the README.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Substrings that would reintroduce a status image or a package-index URL.
_BANNED_URL_MARKERS = ("badge.svg", "img.shields.io", "pypi.org")
# Author-machine prefixes, split so this file does not contain the path itself.
_ABS_PATHS = ("/home" + "/", "/Users" + "/", "C:" + "\\Users\\")


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _ci() -> str:
    return (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")


def test_readme_count_lines_match_the_tree():
    env = os.environ.copy()
    src = str(ROOT / "src")
    prev = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not prev else src + os.pathsep + prev
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    collected = proc.stdout + proc.stderr
    match = re.search(r"(\d+) tests collected", collected)
    assert match, collected
    n_tests = int(match.group(1))

    ops = (ROOT / "src/vlab/ops.py").read_text(encoding="utf-8")
    n_ops = len(re.findall(r"^@torch\.library\.impl\(", ops, re.M))
    gate = (ROOT / "tests/mutation_gate.py").read_text(encoding="utf-8")
    n_mut = len(re.findall(r'^    "[A-Z0-9_]+": \(', gate, re.M))

    readme = _readme()
    for line in (
        f"Collected pytest tests: {n_tests}",
        f"Registered aten impls: {n_ops}",
        f"Mutation injections: {n_mut}",
    ):
        assert line in readme, line


def test_readme_quotes_the_ci_commands():
    readme = _readme()
    ci = _ci()
    shared = (
        'pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6"',
        "pip install pytest numpy",
        "pip install -e .",
        "pytest tests/ -q",
        "python tests/mutation_gate.py",
    )
    for line in shared:
        assert line in ci, line
        assert line in readme, line
    assert '"3.10"' in ci and '"3.12"' in ci
    assert "3.10" in readme and "3.12" in readme
    assert "_REGISTERED" in ci
    assert "mutation_gate.py" in readme


def test_docs_do_not_carry_machine_paths_badges_or_a_pinned_torch_patch():
    readme = _readme()
    failure_modes = (ROOT / "docs/FAILURE_MODES.md").read_text(encoding="utf-8")
    vendor = (ROOT / "docs/vendor-hooks.md").read_text(encoding="utf-8")
    blob = "\n".join((readme, failure_modes, vendor, _ci()))
    for marker in _BANNED_URL_MARKERS:
        assert marker not in blob, marker
    assert "2.14.1" not in readme
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", ".pytest_cache", "build", "dist"} or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.suffix in {".pyc", ".png", ".whl"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for needle in _ABS_PATHS:
            assert needle not in text, f"{path} contains an author-machine path"
