from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass
class CacheRepository:
    db_path: Path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS listen_history (
                    track_id TEXT NOT NULL,
                    played_at TEXT NOT NULL,
                    PRIMARY KEY (track_id, played_at)
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lifetime_counts (
                    track_id TEXT PRIMARY KEY,
                    play_count INTEGER NOT NULL DEFAULT 0
                );
                """
            )

    def record_listens(self, recent_df: pd.DataFrame) -> None:
        if recent_df.empty:
            return
        with self._connect() as conn:
            rows = recent_df[["track_id", "played_at"]].dropna().values.tolist()
            conn.executemany(
                "INSERT OR IGNORE INTO listen_history(track_id, played_at) VALUES(?, ?);",
                rows,
            )
            # Update lifetime counts
            counts: Dict[str, int] = {}
            for track_id in recent_df["track_id"].dropna().tolist():
                counts[track_id] = counts.get(track_id, 0) + 1
            for track_id, inc in counts.items():
                conn.execute(
                    "INSERT INTO lifetime_counts(track_id, play_count) VALUES(?, ?) ON CONFLICT(track_id) DO UPDATE SET play_count = play_count + excluded.play_count;",
                    (track_id, inc),
                )

    def get_recent_history(self, days: int = 14) -> pd.DataFrame:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT track_id, played_at FROM listen_history WHERE played_at >= ? ORDER BY played_at DESC;",
                conn,
                params=(cutoff,),
            )
        return df

    def get_lifetime_counts(self) -> Dict[str, int]:
        with self._connect() as conn:
            df = pd.read_sql_query("SELECT track_id, play_count FROM lifetime_counts;", conn)
        return {r["track_id"]: int(r["play_count"]) for _, r in df.iterrows()} if not df.empty else {}
