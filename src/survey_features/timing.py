"""
Wall-clock timing for pipeline phases and nested spans.

Append-only JSONL (same spirit as llm.TokenUsageLog) plus in-memory aggregates
for an end-of-phase stdout summary. Safe for concurrent ThreadPool workers.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import OUTPUTS_DIR


def resolve_workers(cli_value: int | None, env_name: str, default: int = 1) -> int:
    """CLI > env > default. Always >= 1."""
    if cli_value is not None:
        return max(1, int(cli_value))
    raw = os.environ.get(env_name)
    if raw is not None and raw.strip():
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, int(default))


def default_timing_path(phase: str, tag: str | None = None) -> Path:
    """outputs/logs/timing_<phase>[_<tag>]_<UTC stamp>.jsonl"""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = ["timing", phase]
    if tag:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)[:64]
        parts.append(safe)
    parts.append(stamp)
    return OUTPUTS_DIR / "logs" / ("_".join(parts) + ".jsonl")


@dataclass
class TimingLog:
    """Append-only JSONL timing log with per-name aggregates."""

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _agg: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, name: str, wall_s: float, **extra: Any) -> None:
        row: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "wall_s": round(float(wall_s), 6),
            **{k: v for k, v in extra.items() if v is not None},
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            g = self._agg.setdefault(
                name, {"n": 0.0, "total_s": 0.0, "max_s": 0.0, "min_s": float("inf")}
            )
            g["n"] += 1
            g["total_s"] += float(wall_s)
            g["max_s"] = max(g["max_s"], float(wall_s))
            g["min_s"] = min(g["min_s"], float(wall_s))

    @contextmanager
    def span(self, name: str, **extra: Any) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - t0, **extra)

    def print_summary(self, prefix: str = "[timing]") -> None:
        if not self._agg:
            print(f"{prefix} no spans recorded -> {self.path.resolve()}")
            return
        print(f"{prefix} JSONL: {self.path.resolve()}")
        for name in sorted(self._agg):
            g = self._agg[name]
            n = int(g["n"])
            total = g["total_s"]
            mean = total / n if n else 0.0
            mn = g["min_s"] if g["min_s"] != float("inf") else 0.0
            print(
                f"  {name:32s}  n={n:5d}  "
                f"total={total:8.1f}s  mean={mean:6.2f}s  "
                f"min={mn:6.2f}s  max={g['max_s']:6.2f}s"
            )
