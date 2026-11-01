"""Run the FiQA RAG FastAPI service with Uvicorn."""

import argparse
import os
import sys

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

load_dotenv(os.path.join(_REPO_ROOT, ".env"))


def listening_port(cli_port: int | None) -> int:
    """CLI ``--port`` wins; else ``PORT`` env (Cloud Run); else 8000 for local dev."""
    if cli_port is not None:
        return cli_port
    return int(os.environ.get("PORT", "8000"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve FiQA RAG API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Listen port (default: PORT env or 8000)",
    )
    parser.add_argument("--reload", action="store_true", help="Reload on code changes (dev)")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=args.host,
        port=listening_port(args.port),
        reload=args.reload,
        app_dir=os.path.join(_REPO_ROOT, "src"),
    )


if __name__ == "__main__":
    main()
