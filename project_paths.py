"""Shared path helpers for repo-local execution."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def repo_path(*parts: str) -> Path:
    """Build a path relative to the repository root."""
    return REPO_ROOT.joinpath(*parts)


def path_from_env(name: str, default: Path) -> Path:
    """Return a Path from an environment variable, falling back to default."""
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default


def str_from_env(name: str, default: str) -> str:
    """Return a string from an environment variable, falling back to default."""
    return os.getenv(name, default)


ABAQUS_CMD = str_from_env("ABAQUS_CMD", r"C:\SIMULIA\Commands\abaqus.bat")
RESULTS_ROOT = path_from_env("UQ_RESULTS_ROOT", repo_path("results"))
