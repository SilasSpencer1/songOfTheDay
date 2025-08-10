from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from song_of_the_day.config import Settings, load_settings


TABLE_SQL = """
CREATE TABLE IF NOT EXISTS daily_picks (
    user_id TEXT NOT NULL,
    day_utc TEXT NOT NULL,
    track_id TEXT NOT NULL,
    name TEXT,
    artist TEXT,
    preview_url TEXT,
    rationale TEXT,
    alternates_json TEXT NOT NULL,
    PRIMARY KEY (user_id, day_utc)
);
"""


@dataclass
class DailyPickStore:
    db_path: Path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def ensure(self) -> None:
        with self._connect() as conn:
            conn.execute(TABLE_SQL)

    def get(self, user_id: str, day_utc: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT track_id, name, artist, preview_url, rationale, alternates_json FROM daily_picks WHERE user_id=? AND day_utc=?;",
                (user_id, day_utc),
            ).fetchone()
            if not row:
                return None
            track_id, name, artist, preview_url, rationale, alternates_json = row
            alternates = json.loads(alternates_json) if alternates_json else []
            return {
                "song_of_day": {
                    "id": track_id,
                    "name": name,
                    "artist": artist,
                    "preview_url": preview_url,
                    "rationale": rationale,
                },
                "alternates": alternates,
            }

    def put(self, user_id: str, day_utc: str, pick: Dict, alternates: List[Dict]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO daily_picks(user_id, day_utc, track_id, name, artist, preview_url, rationale, alternates_json)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, day_utc)
                DO UPDATE SET track_id=excluded.track_id, name=excluded.name, artist=excluded.artist,
                              preview_url=excluded.preview_url, rationale=excluded.rationale,
                              alternates_json=excluded.alternates_json;
                """,
                (
                    user_id,
                    day_utc,
                    pick.get("id"),
                    pick.get("name"),
                    pick.get("artist"),
                    pick.get("preview_url"),
                    pick.get("rationale"),
                    json.dumps(alternates),
                ),
            )


def _deterministic_index(user_id: str, day_utc: str, k: int) -> int:
    if k <= 0:
        return 0
    h = hashlib.sha256(f"{user_id}:{day_utc}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % k


def pick_song_of_day(
    user_id: str,
    ranked_results: List[Dict],
    settings: Optional[Settings] = None,
    utc_day: Optional[str] = None,
) -> Dict:
    """
    ranked_results: list of dicts with keys including:
      - track_id, name/track_name, artist_name/artist_names, preview_url (optional), rationale
    returns dict: { song_of_day: {id, name, artist, preview_url, rationale}, alternates: [5 items] }
    """
    settings = settings or load_settings()
    store = DailyPickStore(settings.db_path)
    store.ensure()

    # Determine day string (UTC)
    day_utc = utc_day or datetime.now(timezone.utc).date().isoformat()

    # Already picked today?
    cached = store.get(user_id, day_utc)
    if cached:
        return cached

    if not ranked_results:
        return {"song_of_the_day": None, "alternates": []}

    # Choose among top-5 deterministically for variety without drifting from top quality
    k_pool = min(5, len(ranked_results))
    idx = _deterministic_index(user_id, day_utc, k_pool)
    chosen = ranked_results[idx]

    def _norm_name(x: Dict, k1: str, k2: str) -> Optional[str]:
        return x.get(k1) or x.get(k2)

    # Build pick payload
    pick = {
        "id": chosen.get("track_id") or chosen.get("id"),
        "name": _norm_name(chosen, "name", "track_name"),
        "artist": _norm_name(chosen, "artist_name", "artist_names"),
        "preview_url": chosen.get("preview_url"),
        "rationale": chosen.get("rationale"),
    }

    # Alternates: top 5 excluding chosen
    alternates: List[Dict] = []
    for r in ranked_results:
        tid = r.get("track_id") or r.get("id")
        if tid == pick["id"]:
            continue
        alternates.append(
            {
                "id": tid,
                "name": _norm_name(r, "name", "track_name"),
                "artist": _norm_name(r, "artist_name", "artist_names"),
                "preview_url": r.get("preview_url"),
                "rationale": r.get("rationale"),
            }
        )
        if len(alternates) >= 5:
            break

    store.put(user_id, day_utc, pick, alternates)

    return {"song_of_day": pick, "alternates": alternates}

