"""Reproducibility helpers for evaluation artifacts and reports."""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    return _REPO_ROOT


def repo_relative_path(path: str | Path, *, root: Path | None = None) -> str:
    """Return a portable forward-slash path relative to the repository root."""
    base = root or _REPO_ROOT
    target = Path(path)
    try:
        if not target.is_absolute():
            return target.as_posix()
        rel = target.resolve().relative_to(base.resolve())
        return rel.as_posix()
    except ValueError:
        return target.as_posix()


def try_git_commit_short() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None
