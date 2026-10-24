"""Smoke-test a running FiQA RAG API (Block 0). Exit 0 on success, 1 on failure."""

from __future__ import annotations

import argparse
import json
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test FiQA RAG HTTP API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Only check /health and /ready",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="After /query, also smoke POST /query/stream (SSE)",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    with httpx.Client(timeout=120.0) as client:
        health = client.get(f"{base}/health", headers={"X-Request-ID": "smoke-week6"})
        if health.status_code != 200:
            print(f"FAIL /health -> {health.status_code}", file=sys.stderr)
            return 1
        if health.headers.get("X-Request-ID") != "smoke-week6":
            print("FAIL /health missing echoed X-Request-ID", file=sys.stderr)
            return 1
        print("OK /health", health.json())

        ready = client.get(f"{base}/ready")
        if ready.status_code == 200:
            print("OK /ready", ready.json())
        else:
            print(f"WARN /ready -> {ready.status_code}", file=sys.stderr)
            try:
                print(json.dumps(ready.json(), indent=2), file=sys.stderr)
            except json.JSONDecodeError:
                print(ready.text, file=sys.stderr)
            if args.skip_query:
                return 1

        if args.skip_query:
            return 0 if ready.status_code == 200 else 1

        if ready.status_code != 200:
            print("SKIP /query (not ready)", file=sys.stderr)
            return 1

        query = client.post(
            f"{base}/query",
            json={"question": "What is APR on a credit card?"},
        )
        if query.status_code != 200:
            print(f"FAIL /query -> {query.status_code}", file=sys.stderr)
            print(query.text, file=sys.stderr)
            return 1
        body = query.json()
        print("OK /query", {"answer_len": len(body.get("answer", "")), "citations": len(body.get("citations", []))})

        if args.stream:
            with client.stream(
                "POST",
                f"{base}/query/stream",
                json={"question": "What is APR on a credit card?"},
            ) as stream_resp:
                if stream_resp.status_code != 200:
                    print(f"FAIL /query/stream -> {stream_resp.status_code}", file=sys.stderr)
                    return 1
                text = stream_resp.read().decode()
            if "event: done" not in text:
                print("FAIL /query/stream: no done event", file=sys.stderr)
                return 1
            print("OK /query/stream (SSE done received)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
