from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NoveltyConfig:
    novel_days_ago: int = 90
    alpha: float = 0.7
    beta: float = 0.4
    w1: float = 0.5
    w2: float = 0.3
    w3: float = 0.2


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def build_user_history_profile(
    recent_df: pd.DataFrame,
    top_tracks_df: pd.DataFrame,
    days_recent_window: int = 30,
    cfg: NoveltyConfig = NoveltyConfig(),
) -> pd.DataFrame:
    recent_df = recent_df.copy() if recent_df is not None else pd.DataFrame()
    top_tracks_df = top_tracks_df.copy() if top_tracks_df is not None else pd.DataFrame()

    # Normalize columns we rely on
    for col in ["track_id", "artist_ids", "played_at", "artist_names", "track_name", "album"]:
        if col not in recent_df.columns:
            recent_df[col] = None

    # Filter recent by window
    cutoff = datetime.utcnow() - timedelta(days=days_recent_window)
    if not recent_df.empty:
        recent_df["played_dt"] = recent_df["played_at"].astype(str).map(_parse_iso)
        recent_df = recent_df[recent_df["played_dt"].notna()]
        recent_df = recent_df[recent_df["played_dt"] >= cutoff]

    # play_count_recent and last_played_at
    if recent_df.empty:
        agg_recent = pd.DataFrame(columns=["track_id", "play_count_recent", "last_played_at", "artist_id"])
    else:
        # choose primary artist as first in list
        recent_df["artist_id"] = recent_df["artist_ids"].apply(lambda xs: xs[0] if isinstance(xs, list) and xs else None)
        agg_recent = (
            recent_df.groupby(["track_id", "artist_id"], dropna=False)
            .agg(
                play_count_recent=("track_id", "count"),
                last_played_at=("played_dt", "max"),
            )
            .reset_index()
        )

    # approximate total_play_count: recent count + whether appears in tops list
    tops_tracks = set()
    if not top_tracks_df.empty:
        if "track_id" in top_tracks_df.columns:
            tops_tracks = set(top_tracks_df["track_id"].dropna().tolist())
        elif "id" in top_tracks_df.columns:
            tops_tracks = set(top_tracks_df["id"].dropna().tolist())

    agg_recent["total_play_count"] = agg_recent.get("play_count_recent", 0) + agg_recent["track_id"].isin(tops_tracks).astype(int)

    # artist_play_count within recent window
    if not agg_recent.empty:
        art_counts = agg_recent.groupby("artist_id")["play_count_recent"].sum().to_dict()
        agg_recent["artist_play_count"] = agg_recent["artist_id"].map(lambda a: int(art_counts.get(a, 0)))
    else:
        agg_recent["artist_play_count"] = 0

    # Exclude tracks with play_count_recent >= 3 in last window
    agg_recent = agg_recent[agg_recent["play_count_recent"].fillna(0) < 3]

    # Compute novelty
    if agg_recent.empty:
        # return empty with expected columns
        return pd.DataFrame(columns=[
            "track_id",
            "artist_id",
            "novelty",
            "play_count_recent",
            "last_played_at",
            "total_play_count",
            "artist_play_count",
        ])

    novel_cutoff = datetime.utcnow() - timedelta(days=cfg.novel_days_ago)
    indicator_novel = agg_recent["last_played_at"].map(lambda dt: 1.0 if (dt is None or dt > novel_cutoff) else 0.0)

    novelty = (
        cfg.w1 * np.exp(-cfg.alpha * agg_recent["play_count_recent"].fillna(0))
        + cfg.w2 * indicator_novel
        + cfg.w3 * np.exp(-cfg.beta * agg_recent["artist_play_count"].fillna(0))
    )

    result = agg_recent.copy()
    result["novelty"] = novelty.astype(float)

    # joinable metadata: bring back simple names from recent if available
    meta_cols = ["track_name", "artist_names", "album"]
    meta_df = recent_df.drop_duplicates("track_id")[["track_id"] + [c for c in meta_cols if c in recent_df.columns]] if not recent_df.empty else pd.DataFrame(columns=["track_id"])
    result = result.merge(meta_df, on="track_id", how="left")

    return result[[
        "track_id",
        "artist_id",
        "novelty",
        "play_count_recent",
        "last_played_at",
        "total_play_count",
        "artist_play_count",
        *[c for c in ["track_name", "artist_names", "album"] if c in result.columns],
    ]].reset_index(drop=True)
