"""OpenRouter LLM client with caching, logging and retries.

Every call is logged as JSONL (prompt, raw response, model, timestamp, seed,
condition, wording, row, purpose) and cached by request hash so reruns cost
nothing (SPEC §1.6). Temperature 1.0, no tuning (SPEC §1.3).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class CallResult:
    content: str
    model_version: str
    cached: bool
    usage: dict[str, int] = field(default_factory=dict)
    attempts: int = 1


class LLMClient:
    """Thin wrapper around OpenRouter with a JSONL cache.

    Parameters
    ----------
    cache_path : JSONL file; one line per call. A call is a cache hit when a
        line with the same request hash exists (then the stored response is
        returned and no API call is made).
    log_path : JSONL audit log; every call (hit or miss) is appended here with
        its metadata. Separate from the cache so the cache can be deleted.
    """

    def __init__(
        self,
        cache_path: str | Path,
        log_path: str | Path,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_workers: int = 16,
    ):
        self.cache_path = Path(cache_path)
        self.log_path = Path(log_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set (check .env)")
        self._client = OpenAI(api_key=key, base_url=base_url, timeout=300)
        self._cache = _RequestCache(self.cache_path)
        self._log_lock = threading.Lock()
        self.max_workers = max_workers

    @staticmethod
    def request_hash(
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_completion_tokens: int,
        response_format: dict[str, str] | None,
        seed: int | None,
    ) -> str:
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_completion_tokens": max_completion_tokens,
                "response_format": response_format,
                "seed": seed,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        meta: dict[str, Any] | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 4000,
        json_mode: bool = True,
        seed: int | None = None,
        max_retries: int = 5,
        purpose: str = "",
    ) -> CallResult:
        """One chat completion, served from cache when possible.

        ``seed`` is a *label* recorded with the call (draw seeds live outside
        the API): setting a different seed forces a fresh API call for the
        same prompt, which is how independent runs redo their calls.
        """
        seed = int(seed) if seed is not None else None
        response_format = {"type": "json_object"} if json_mode else None
        req_hash = self.request_hash(
            model, messages, temperature, top_p, max_completion_tokens, response_format, seed
        )
        meta = dict(meta or {})
        meta.setdefault("purpose", purpose)

        hit = self._cache.get(req_hash)
        if hit is not None:
            self._log(hit["content"], model, meta, cached=True, req_hash=req_hash)
            return CallResult(
                content=hit["content"],
                model_version=hit.get("model_version", ""),
                cached=True,
                usage=hit.get("usage", {}),
            )

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                kwargs: dict[str, Any] = dict(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_completion_tokens=max_completion_tokens,
                )
                if response_format is not None:
                    kwargs["response_format"] = response_format
                resp = self._client.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    raise ValueError("empty completion")
                usage = {}
                if resp.usage is not None:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens or 0,
                        "completion_tokens": resp.usage.completion_tokens or 0,
                        "total_tokens": resp.usage.total_tokens or 0,
                    }
                model_version = getattr(resp, "model", model) or model
                self._cache.put(
                    req_hash,
                    {
                        "content": content,
                        "model_version": model_version,
                        "usage": usage,
                    },
                )
                self._log(content, model, meta, cached=False, req_hash=req_hash,
                          usage=usage, model_version=model_version, attempt=attempt)
                return CallResult(
                    content=content, model_version=model_version, cached=False, usage=usage
                )
            except Exception as e:  # noqa: BLE001 - network/parse errors all retryable
                last_err = e
                wait = min(2 ** attempt, 30)
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")

    def _log(
        self,
        content: str,
        model: str,
        meta: dict[str, Any],
        *,
        cached: bool,
        req_hash: str,
        usage: dict[str, int] | None = None,
        model_version: str = "",
        attempt: int = 0,
    ) -> None:
        record = {
            "ts": _utcnow(),
            "req_hash": req_hash,
            "model": model,
            "model_version": model_version,
            "cached": cached,
            "attempt": attempt,
            "usage": usage or {},
            **meta,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._log_lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


class _RequestCache:
    """Append-only JSONL cache keyed by request hash."""

    def __init__(self, path: Path):
        self.path = path
        self._map: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._map[rec["req_hash"]] = rec

    def get(self, req_hash: str) -> dict[str, Any] | None:
        with self._lock:
            return self._map.get(req_hash)

    def put(self, req_hash: str, payload: dict[str, Any]) -> None:
        rec = {"req_hash": req_hash, "ts": _utcnow(), **payload}
        with self._lock:
            self._map[req_hash] = rec
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_parallel(fn, items, max_workers: int = 8, desc: str = "") -> list:
    """Map fn(index, item) over items preserving order; failures re-raised.

    The cache and log are thread-safe, so this is safe for LLM calls.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[Any] = [None] * len(items)
    if not items:
        return results
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as ex:
        futs = {ex.submit(fn, i, it): i for i, it in enumerate(items)}
        done = 0
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            if desc and done % 50 == 0:
                print(f"[{desc}] {done}/{len(items)}", flush=True)
    return results
