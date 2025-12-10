from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _cache_dir() -> Path:
    return Path.home() / ".cache" / "vllm_autoconfig"


def make_cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def load_cached_plan(key: str) -> dict[str, Any] | None:
    path = _cache_dir() / "plans" / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_cached_plan(key: str, plan: dict[str, Any]) -> None:
    plans_dir = _cache_dir() / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    path = plans_dir / f"{key}.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
