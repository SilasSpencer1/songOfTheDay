from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def compute_novelty_scores(
    candidate_ids: Iterable[str],
    recent_history: pd.DataFrame,
    lifetime_counts: Dict[str, int],
    now_utc: datetime | None = None,
    alpha: float = 0.15,
) -> Dict[str, float]:
    now = now_utc or datetime.utcnow()

    def recency_factor(last_played_at_str: str | None) -> float:
        if not last_played_at_str:
            return 1.0
        try:
            last_dt = datetime.fromisoformat(last_played_at_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return 1.0
        days = (now - last_dt).days
        if days <= 3:
            return 0.0
        if days <= 7:
            return 0.3
        if days <= 14:
            return 0.6
        return 1.0

    last_play_map: Dict[str, str] = {}
    if not recent_history.empty:
        recent_history_sorted = recent_history.sort_values("played_at", ascending=False)
        last_play_map = (
            recent_history_sorted.drop_duplicates("track_id")[["track_id", "played_at"]]
            .set_index("track_id")["played_at"]
            .to_dict()
        )

    scores: Dict[str, float] = {}
    for track_id in candidate_ids:
        last_played = last_play_map.get(track_id)
        rec_factor = recency_factor(last_played)
        lifetime = lifetime_counts.get(track_id, 0)
        lifetime_factor = float(np.exp(-alpha * lifetime))
        scores[track_id] = float(rec_factor * lifetime_factor)
    return scores
