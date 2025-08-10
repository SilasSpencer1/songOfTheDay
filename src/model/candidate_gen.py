from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from model.history import build_user_history_profile, NoveltyConfig
from model.mood import mood_to_feature_targets


@dataclass(frozen=True)
class CandidateGenConfig:
    recent_days_window: int = 30
    saved_days_window: int = 60
    target_total: int = 200


def _primary_artist_id(artist_ids: Any) -> Optional[str]:
    if isinstance(artist_ids, list) and artist_ids:
        return artist_ids[0]
    return None


def _get_sp(client: Any):
    sp = getattr(client, "sp", None)
    if sp is None:
        sp = getattr(client, "_sp", None)
    return sp


def _fetch_recent_df(client: Any, days: int) -> pd.DataFrame:
    # data client supports days_back; legacy path uses limit only
    if hasattr(client, "get_recent_tracks"):
        try:
            return client.get_recent_tracks(limit=200, days_back=days)
        except TypeError:
            return client.get_recent_tracks(limit=50)  # type: ignore[arg-type]
    return pd.DataFrame()


def _fetch_top_artists_df(client: Any) -> pd.DataFrame:
    if hasattr(client, "get_top_tracks_and_artists"):
        tops = client.get_top_tracks_and_artists(range="medium_term", limit=50)
        return tops.get("artists", pd.DataFrame())
    if hasattr(client, "get_top_artists"):
        return client.get_top_artists(limit=50, time_range="medium_term")
    return pd.DataFrame()


def _fetch_saved_recent_track_ids(client: Any, days: int) -> set[str]:
    # Optional filter, only if raw spotipy is available
    sp = _get_sp(client)
    if sp is None:
        return set()
    cutoff = datetime.utcnow() - timedelta(days=days)
    track_ids: set[str] = set()
    offset = 0
    page_size = 50
    while True:
        payload = sp.current_user_saved_tracks(limit=page_size, offset=offset)
        items = payload.get("items", [])
        if not items:
            break
        for it in items:
            added_at = it.get("added_at")
            dt = None
            try:
                if added_at:
                    dt = datetime.fromisoformat(added_at.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                dt = None
            if dt and dt >= cutoff:
                tr = it.get("track", {})
                tid = tr.get("id")
                if tid:
                    track_ids.add(tid)
        # Stop early if this page is entirely older than cutoff to reduce paging
        if items and all((datetime.fromisoformat(x.get("added_at").replace("Z", "+00:00")).replace(tzinfo=None) < cutoff) for x in items if x.get("added_at")):
            break
        offset += len(items)
        if len(items) < page_size:
            break
    return track_ids


def _artist_genres_map(client: Any, artist_ids: Iterable[str]) -> Dict[str, List[str]]:
    artist_ids = [a for a in set(artist_ids) if a]
    if not artist_ids:
        return {}
    if hasattr(client, "get_artist_genres"):
        return client.get_artist_genres(artist_ids)  # type: ignore[no-any-return]
    # fallback to raw spotipy
    sp = _get_sp(client)
    if sp is None:
        return {}
    genres_map: Dict[str, List[str]] = {}
    for i in range(0, len(artist_ids), 50):
        batch = artist_ids[i : i + 50]
        artists = sp.artists(batch).get("artists", [])
        for a in artists:
            genres_map[a.get("id")] = a.get("genres", [])
    return genres_map


def _available_genre_seeds(client: Any) -> set[str]:
    sp = _get_sp(client)
    if sp is None:
        return set()
    try:
        seeds = sp.recommendations_available_genre_seeds().get("genres", [])
        return set(seeds)
    except Exception:
        return set()


def _normalize_seed_genres(client: Any, raw_genres: List[str]) -> List[str]:
    if not raw_genres:
        return []
    available = _available_genre_seeds(client)
    if not available:
        # Fallback: naive normalization; keep hyphens
        def naive(g: str) -> str:
            g = g.lower().strip()
            g = g.replace("&", " and ")
            g = "-".join(g.split())
            return g
        seeds = [naive(g) for g in raw_genres]
        # unique, keep order
        seen = set()
        out = []
        for s in seeds:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out[:10]

    # Build token sets for fuzzy match
    def tok(s: str) -> set[str]:
        return set(s.replace("&", " and ").replace("/", " ").replace("+", " ").replace("_", " ").replace("-", " ").split())

    canon = {g: tok(g) for g in available}

    mapped: List[str] = []
    seen: set[str] = set()
    for g in raw_genres:
        gg = g.lower().strip()
        # direct normalized attempt
        guess = gg.replace("&", " and ")
        guess = "-".join(guess.split())
        if guess in available and guess not in seen:
            seen.add(guess)
            mapped.append(guess)
            continue
        # special-case common Spotify name
        special = {
            "r&b": "r-n-b",
            "rnb": "r-n-b",
            "alt r&b": "r-n-b",
            "lo fi": "lo-fi",
            "lo-fi indie": "lo-fi",
            "bedroom pop": "bedroom-pop",
            "hip hop": "hip-hop",
            "k pop": "k-pop",
            "j pop": "j-pop",
        }
        if gg in special and special[gg] in available and special[gg] not in seen:
            seen.add(special[gg])
            mapped.append(special[gg])
            continue
        # fuzzy by token overlap
        gtoks = tok(gg)
        best = None
        best_score = 0
        for a, atoks in canon.items():
            score = len(gtoks & atoks)
            if score > best_score:
                best_score = score
                best = a
        if best and best not in seen:
            seen.add(best)
            mapped.append(best)

    return mapped[:10]


def _targets_from_mood(mood: Optional[str]) -> Optional[Dict[str, float]]:
    if not mood:
        return None
    ranges = mood_to_feature_targets(mood)
    # Convert ranges to midpoints for Spotify target_* params
    targets = {feat: float((lo + hi) / 2.0) for feat, (lo, hi) in ranges.items()}
    return targets


def _fallback_artist_top_tracks(client: Any, seed_artists: List[str], limit: int) -> pd.DataFrame:
    sp = _get_sp(client)
    if sp is None or not seed_artists:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    per = max(5, int(limit / max(1, len(seed_artists))))
    for a in seed_artists[:5]:
        try:
            top = sp.artist_top_tracks(a, country="US").get("tracks", [])
            for t in top[:per]:
                rows.append({
                    "track_id": t.get("id"),
                    "track_name": t.get("name"),
                    "artist_ids": [x.get("id") for x in t.get("artists", [])],
                    "artist_names": ", ".join(x.get("name") for x in t.get("artists", [])),
                    "album": t.get("album", {}).get("name"),
                })
        except Exception:
            continue
    return pd.DataFrame(rows).drop_duplicates("track_id").head(limit)


def _fetch_candidates(client: Any, seed_artists: List[str], seed_genres: List[str], targets: Optional[Dict[str, float]], total: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    per_call = min(100, total)
    # make two calls with different slices to diversify
    slices: List[Tuple[List[str], List[str]]] = [
        (seed_artists[:5], seed_genres[:5]),
        (seed_artists[-5:], seed_genres[-5:]) if len(seed_artists) > 5 or len(seed_genres) > 5 else (seed_artists[:5], seed_genres[:5]),
    ]
    for sa, sg in slices:
        try:
            if hasattr(client, "search_candidates"):
                df = client.search_candidates(seed_artists=sa or None, seed_genres=(sg or None), audio_feature_targets=targets, limit=per_call)
            else:
                # legacy recommendations path
                df = client.get_recommendations(seed_artists=sa or None, seed_genres=(sg or None), seed_tracks=None, limit=per_call, target_features=targets)  # type: ignore[attr-defined]
        except Exception:
            # Retry without genres if genre seeds invalid
            try:
                if hasattr(client, "search_candidates"):
                    df = client.search_candidates(seed_artists=sa or None, seed_genres=None, audio_feature_targets=targets, limit=per_call)
                else:
                    df = client.get_recommendations(seed_artists=sa or None, seed_genres=None, seed_tracks=None, limit=per_call, target_features=targets)  # type: ignore[attr-defined]
            except Exception:
                df = pd.DataFrame()
        frames.append(df)
        if sum(len(f) for f in frames) >= total:
            break

    filtered = [f for f in frames if f is not None and not f.empty]
    out = pd.concat(filtered, ignore_index=True) if filtered else pd.DataFrame()

    # Final fallback: artist top-tracks if recommendations are empty
    if out.empty:
        out = _fallback_artist_top_tracks(client, seed_artists, total)

    return out.drop_duplicates("track_id").head(total)


def generate_candidate_pool(client: Any, mood: Optional[str] = None, cfg: CandidateGenConfig = CandidateGenConfig()) -> pd.DataFrame:
    # User signals
    recent_df = _fetch_recent_df(client, cfg.recent_days_window)
    top_artists_df = _fetch_top_artists_df(client)

    # Compute recent counts per track and per artist to downweight overplayed
    recent_df = recent_df.copy() if not recent_df.empty else pd.DataFrame(columns=["track_id", "artist_ids", "played_at"])
    if not recent_df.empty and "artist_id" not in recent_df.columns:
        recent_df["artist_id"] = recent_df["artist_ids"].apply(_primary_artist_id)

    track_recent_counts = recent_df.groupby("track_id").size().to_dict() if not recent_df.empty else {}
    overplayed_track_ids = {tid for tid, c in track_recent_counts.items() if c >= 3}

    artist_recent_counts = recent_df.groupby("artist_id").size().to_dict() if not recent_df.empty else {}

    # Seeds: top artists downweighted by recent play count
    if not top_artists_df.empty:
        # ensure columns
        if "artist_id" not in top_artists_df.columns and "id" in top_artists_df.columns:
            top_artists_df = top_artists_df.rename(columns={"id": "artist_id", "name": "artist_name"})
        top_artists_df["recent_play_count"] = top_artists_df["artist_id"].map(lambda a: int(artist_recent_counts.get(a, 0)))
        top_artists_df = top_artists_df.sort_values(["recent_play_count", "popularity"] if "popularity" in top_artists_df.columns else ["recent_play_count"], ascending=[True, False])
        seed_artists = top_artists_df["artist_id"].dropna().head(10).tolist()
    else:
        seed_artists = []

    # Seed genres from top artists genres
    genre_counts: Counter[str] = Counter()
    if not top_artists_df.empty and "genres" in top_artists_df.columns:
        for gs in top_artists_df["genres"].tolist():
            if isinstance(gs, list):
                genre_counts.update(gs)
    raw_seed_genres = [g for g, _ in genre_counts.most_common(10)]
    seed_genres = _normalize_seed_genres(client, raw_seed_genres)

    # Mood targets
    targets = _targets_from_mood(mood)

    # Fetch candidates
    raw_cands = _fetch_candidates(client, seed_artists, seed_genres, targets, cfg.target_total)

    if raw_cands.empty:
        return pd.DataFrame(columns=["track_id", "artist_id", "name", "artist_name", "genres", "novelty"])  # minimal schema

    # Normalize columns
    raw = raw_cands.copy()
    if "artist_ids" not in raw.columns:
        raw["artist_ids"] = [[] for _ in range(len(raw))]
    raw["artist_id"] = raw["artist_ids"].apply(_primary_artist_id)
    raw["name"] = raw.get("track_name", raw.get("name", "")).astype(str)
    raw["artist_name"] = raw.get("artist_names", raw.get("artist_name", "")).astype(str)

    # Filter: recent overplayed
    raw = raw[~raw["track_id"].isin(overplayed_track_ids)]

    # Filter: saved within last N days (if available)
    saved_recent_ids = _fetch_saved_recent_track_ids(client, cfg.saved_days_window)
    if saved_recent_ids:
        raw = raw[~raw["track_id"].isin(saved_recent_ids)]

    # Add genres for candidate artists
    genres_map = _artist_genres_map(client, raw["artist_id"].dropna().unique().tolist())
    raw["genres"] = raw["artist_id"].map(lambda a: genres_map.get(a, []))

    # Audio features
    if hasattr(client, "get_audio_features"):
        feats = client.get_audio_features(raw["track_id"].dropna().tolist())
    else:
        feats = pd.DataFrame()
    if not feats.empty:
        if "track_id" not in feats.columns and "id" in feats.columns:
            feats = feats.rename(columns={"id": "track_id"})
        raw = raw.merge(feats, on="track_id", how="left")

    # Novelty merge using user history
    top_tracks_df = pd.DataFrame()
    if hasattr(client, "get_top_tracks_and_artists"):
        top_tracks_df = client.get_top_tracks_and_artists(range="medium_term", limit=50).get("tracks", pd.DataFrame())

    history_df = build_user_history_profile(recent_df=recent_df, top_tracks_df=top_tracks_df, days_recent_window=cfg.recent_days_window)
    history_novel_map = history_df.set_index("track_id")["novelty"].to_dict() if not history_df.empty else {}

    # Fallback novelty for unseen tracks: use artist recent counts
    cfg_n = NoveltyConfig()
    def _novelty_fallback(aid: Optional[str]) -> float:
        artist_c = int(history_df.groupby("artist_id")["play_count_recent"].sum().to_dict().get(aid, 0)) if not history_df.empty else int(artist_recent_counts.get(aid, 0))
        return float(cfg_n.w1 * np.exp(-cfg_n.alpha * 0) + cfg_n.w2 * 1.0 + cfg_n.w3 * np.exp(-cfg_n.beta * artist_c))

    raw["novelty"] = raw.apply(lambda r: float(history_novel_map.get(r["track_id"])) if r["track_id"] in history_novel_map else _novelty_fallback(r["artist_id"]), axis=1)

    # Final columns ordering: core + audio features if present
    core_cols = ["track_id", "artist_id", "name", "artist_name", "genres", "novelty"]
    audio_cols = [c for c in [
        "danceability", "energy", "valence", "acousticness", "instrumentalness", "liveness", "speechiness", "tempo", "loudness"
    ] if c in raw.columns]

    cols = core_cols + audio_cols
    return raw[cols].reset_index(drop=True)
