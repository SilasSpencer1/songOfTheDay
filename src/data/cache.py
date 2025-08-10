from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class TTLCache:
    db_path: Path

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_cache (
                    user_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, endpoint, params_hash)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    @staticmethod
    def _hash_params(params: dict) -> str:
        return json.dumps(params, sort_keys=True, separators=(",", ":"))

    def get(self, user_id: str, endpoint: str, params: dict) -> Optional[Any]:
        params_hash = self._hash_params(params)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM api_cache WHERE user_id=? AND endpoint=? AND params_hash=?;",
                (user_id, endpoint, params_hash),
            ).fetchone()
            if not row:
                return None
            payload, expires_at = row
            if expires_at <= now:
                conn.execute(
                    "DELETE FROM api_cache WHERE user_id=? AND endpoint=? AND params_hash=?;",
                    (user_id, endpoint, params_hash),
                )
                return None
            try:
                return json.loads(payload)
            except Exception:
                return None

    def set(self, user_id: str, endpoint: str, params: dict, payload: Any, ttl_seconds: int) -> None:
        params_hash = self._hash_params(params)
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
        payload_str = json.dumps(payload, separators=(",", ":"))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_cache(user_id, endpoint, params_hash, payload, expires_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(user_id, endpoint, params_hash)
                DO UPDATE SET payload=excluded.payload, expires_at=excluded.expires_at;
                """,
                (user_id, endpoint, params_hash, payload_str, expires_at),
            )
