"""Low-level HTTP helpers — zero external dependencies (pure stdlib)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Generator, Iterator, Optional


class VortElioError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"HTTP {status}: {message}")


def _request(
    method: str,
    url: str,
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        try:
            msg = json.loads(exc.read()).get("error", exc.reason)
        except Exception:
            msg = exc.reason
        raise VortElioError(exc.code, msg) from exc


def _request_bytes(
    method: str,
    url: str,
    body: Optional[Dict[str, Any]] = None,
    timeout: int = 300,
) -> bytes:
    """Like _request but returns the raw response body (for binary endpoints)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        try:
            msg = json.loads(exc.read()).get("error", exc.reason)
        except Exception:
            msg = exc.reason
        raise VortElioError(exc.code, msg) from exc


def _stream_ndjson(
    url: str,
    body: Dict[str, Any],
    timeout: int = 300,
) -> Generator[Dict[str, Any], None, None]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def _stream_sse(
    url: str,
    body: Dict[str, Any],
    timeout: int = 300,
) -> Generator[Dict[str, Any], None, None]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        event_type = "message"
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    obj = json.loads(payload)
                    obj.setdefault("_event", event_type)
                    yield obj
                except json.JSONDecodeError:
                    pass
                event_type = "message"
