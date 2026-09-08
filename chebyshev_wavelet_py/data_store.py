"""Structured decision persistence and deterministic replay interfaces.

CSV uses only the Python standard library.  Parquet support is optional and
requires ``pyarrow`` at runtime; no data-science dependency is imposed on the
core trading package merely for importing this module.
"""

from __future__ import annotations

import asyncio
import csv
import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping


StorageFormat = Literal["csv", "parquet"]


@dataclass(frozen=True)
class DecisionRecord:
    """Portable representation of one evaluated instrument decision."""

    timestamp: str
    run_id: str
    instrument: str
    signal: float
    score: float
    expected_edge_bps: float
    required_edge_bps: float
    projected_turnover: float
    approved: bool
    submitted: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_orchestrator_decision(
        cls, decision: Mapping[str, Any], run_id: str | None = None
    ) -> "DecisionRecord":
        """Convert a ``LiveOrchestrator`` decision dictionary into a record."""
        known = {
            "instrument", "signal", "score", "expected_edge_bps", "required_edge_bps",
            "projected_turnover", "approved", "submitted",
        }
        metadata = {key: value for key, value in decision.items() if key not in known}
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=run_id or uuid.uuid4().hex,
            instrument=str(decision["instrument"]),
            signal=float(decision["signal"]),
            score=float(decision["score"]),
            expected_edge_bps=float(decision["expected_edge_bps"]),
            required_edge_bps=float(decision["required_edge_bps"]),
            projected_turnover=float(decision["projected_turnover"]),
            approved=bool(decision["approved"]),
            submitted=bool(decision["submitted"]),
            metadata=metadata,
        )


class DecisionStore:
    """Append structured decisions and replay them in chronological order."""

    _CSV_FIELDS = tuple(DecisionRecord.__dataclass_fields__.keys())

    def __init__(self, path: str | Path, storage_format: StorageFormat = "csv") -> None:
        if storage_format not in {"csv", "parquet"}:
            raise ValueError("storage_format must be 'csv' or 'parquet'")
        self.path = Path(path)
        self.storage_format = storage_format
        self._lock = threading.Lock()

    def append(self, records: Iterable[DecisionRecord]) -> None:
        """Persist records atomically at the record-batch level."""
        batch = list(records)
        if not batch:
            return
        with self._lock:
            if self.storage_format == "csv":
                self._append_csv(batch)
            else:
                self._append_parquet(batch)

    async def append_async(self, records: Iterable[DecisionRecord]) -> None:
        """Persist without blocking an asyncio orchestration loop."""
        batch = list(records)
        await asyncio.to_thread(self.append, batch)

    def iter_records(self) -> Iterator[DecisionRecord]:
        """Yield persisted records in storage order for deterministic replay."""
        if self.storage_format == "csv":
            yield from self._iter_csv()
        else:
            yield from self._iter_parquet()

    def replay(self, handler: Callable[[DecisionRecord], None]) -> int:
        """Replay records through a pure handler and return the record count."""
        count = 0
        for record in self.iter_records():
            handler(record)
            count += 1
        return count

    def _append_csv(self, batch: list[DecisionRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._CSV_FIELDS)
            if write_header:
                writer.writeheader()
            for record in batch:
                row = asdict(record)
                row["metadata"] = json.dumps(row["metadata"], sort_keys=True, default=str)
                writer.writerow(row)

    def _iter_csv(self) -> Iterator[DecisionRecord]:
        if not self.path.exists():
            return
        with self.path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                yield DecisionRecord(
                    timestamp=row["timestamp"],
                    run_id=row["run_id"],
                    instrument=row["instrument"],
                    signal=float(row["signal"]),
                    score=float(row["score"]),
                    expected_edge_bps=float(row["expected_edge_bps"]),
                    required_edge_bps=float(row["required_edge_bps"]),
                    projected_turnover=float(row["projected_turnover"]),
                    approved=_parse_bool(row["approved"]),
                    submitted=_parse_bool(row["submitted"]),
                    metadata=json.loads(row["metadata"] or "{}"),
                )

    def _append_parquet(self, batch: list[DecisionRecord]) -> None:
        pa, pq = _parquet_modules()
        self.path.mkdir(parents=True, exist_ok=True)
        rows = []
        for record in batch:
            row = asdict(record)
            row["metadata"] = json.dumps(row["metadata"], sort_keys=True, default=str)
            rows.append(row)
        table = pa.Table.from_pylist(rows)
        file_path = self.path / f"decision-{uuid.uuid4().hex}.parquet"
        pq.write_table(table, file_path)

    def _iter_parquet(self) -> Iterator[DecisionRecord]:
        _, pq = _parquet_modules()
        if not self.path.exists():
            return
        for file_path in sorted(self.path.glob("*.parquet")):
            for row in pq.read_table(file_path).to_pylist():
                row["metadata"] = json.loads(row.get("metadata") or "{}")
                yield DecisionRecord(**row)


def _parquet_modules() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet storage requires the optional 'pyarrow' package") from exc
    return pa, pq


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


class DataStore:
    """Append JSONL decisions and provide deterministic historical replay.

    JSONL is the operational default because every completed decision is a
    separate durable line and can be inspected after an interrupted process.
    ``DecisionStore`` remains available for CSV and optional Parquet analysis.
    """

    def __init__(self, log_path: str | Path = "logs/decisions.jsonl") -> None:
        self.log_path = Path(log_path)
        self._lock = threading.Lock()

    def append_decision(self, decision: Mapping[str, Any]) -> None:
        """Append one decision as a JSON object, including an UTC timestamp."""
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **dict(decision)}
        encoded = json.dumps(record, sort_keys=True, default=str, separators=(",", ":"))
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()

    async def append_decision_async(self, decision: Mapping[str, Any]) -> None:
        """Append without blocking an asyncio market-data or execution callback."""
        await asyncio.to_thread(self.append_decision, dict(decision))

    def replay(self) -> Iterator[dict[str, Any]]:
        """Yield logged decision objects in recorded order, validating each line."""
        if not self.log_path.exists():
            return
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"malformed JSONL decision at {self.log_path}:{line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise RuntimeError(f"JSONL decision at line {line_number} is not an object")
                yield record

    def replay_to(self, handler: Callable[[Mapping[str, Any]], None]) -> int:
        """Replay decisions through a handler and return the number processed."""
        count = 0
        for record in self.replay():
            handler(record)
            count += 1
        return count
