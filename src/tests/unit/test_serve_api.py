import importlib.util
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "serve_api",
    _REPO_ROOT / "src" / "scripts" / "serve_api.py",
)
assert _spec and _spec.loader
serve_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve_api)
listening_port = serve_api.listening_port


def test_listening_port_defaults_to_8000_without_env():
    os.environ.pop("PORT", None)
    assert listening_port(None) == 8000


def test_listening_port_reads_port_env(monkeypatch):
    monkeypatch.setenv("PORT", "8080")
    assert listening_port(None) == 8080


def test_listening_port_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("PORT", "8080")
    assert listening_port(9000) == 9000
