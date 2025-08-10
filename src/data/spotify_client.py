from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from song_of_the_day.config import load_settings
from .cache import TTLCache


SCOPES = "user-read-recently-played user-top-read playlist-read-private user-library-read"
RECENT_TTL_SECONDS = 6 * 3600
TOPS_TTL_SECONDS = 24 * 3600
AUDIO_TTL_SECONDS = 24 * 3600
RECS_TTL_SECONDS = 3 * 3600


def _retryable_call(fn, *args, max_retries: int = 5, initial_backoff: float = 0.5, **kwargs):
    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except spotipy.SpotifyException as e:
            status = getattr(e, "http_status", None)
            if status == 429:
                retry_after = int(getattr(e, "headers", {}).get("Retry-After", 1))
                time.sleep(max(retry_after, backoff))
            elif status and 500 <= status < 600:
                time.sleep(backoff)
            else:
                raise
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
        backoff = min(backoff * 2, 8.0)


@dataclass
class SpotifyDataClient:
    user_id: str
    cache: TTLCache
    sp: spotipy.Spotify

    @classmethod
    def create(cls) -> "SpotifyDataClient":
        settings = load_settings()
        auth = SpotifyOAuth(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret,
            redirect_uri=settings.spotify_redirect_uri,
            scope=SCOPES,
            open_browser=True,
            cache_path=None,
        )
        sp = spotipy.Spotify(auth_manager=auth)
        me = sp.current_user()
        user_id = me.get("id", "unknown_user")
        cache = TTLCache(settings.db_path)
        return cls(user_id=user_id, cache=cache, sp=sp)

    # recent tracks with optional days_back filter
    def get_recent_tracks(self, limit: int = 200, days_back: int = 30) -> pd.DataFrame:
        params = {"limit": int(limit), "days_back": int(days_back)}
        cached = self.cache.get(self.user_id, "recent_tracks", params)
        if cached is not None:
            return pd.DataFrame(cached)

        items: List[dict] = []
        fetched = 0
        # Spotify max 50 per call for recently played
        remaining = min(limit, 200)
        after_ts = None
        if days_back and days_back > 0:
            cutoff = datetime.utcnow() - timedelta(days=days_back)
            after_ts = int(cutoff.timestamp() * 1000)
        while remaining > 0:
            batch_limit = min(50, remaining)
            payload = _retryable_call(self.sp.current_user_recently_played, limit=batch_limit, after=after_ts)
            page_items = payload.get("items", [])
            if not page_items:
                break
            for it in page_items:
                track = it.get("track", {})
                items.append({
                    "track_id": track.get("id"),
                    "track_name": track.get("name"),
                    "artist_ids": [a.get("id") for a in track.get("artists", [])],
                    "artist_names": ", ".join(a.get("name") for a in track.get("artists", [])),
                    "album": track.get("album", {}).get("name"),
                    "played_at": it.get("played_at"),
                })
            remaining -= len(page_items)
            if after_ts is not None:
                break
            # for before/after paging, use the oldest played_at in this page
            oldest = page_items[-1]["played_at"]
            after_ts = int(datetime.fromisoformat(oldest.replace("Z", "+00:00")).timestamp() * 1000)

        df = pd.DataFrame(items).drop_duplicates("track_id")
        self.cache.set(self.user_id, "recent_tracks", params, df.to_dict(orient="records"), RECENT_TTL_SECONDS)
        return df

    def get_top_tracks_and_artists(self, range: str = "medium_term", limit: int = 50) -> Dict[str, pd.DataFrame]:
        params = {"range": range, "limit": int(limit)}
        cached = self.cache.get(self.user_id, "top_tracks_artists", params)
        if cached is not None:
            return {
                "tracks": pd.DataFrame(cached.get("tracks", [])),
                "artists": pd.DataFrame(cached.get("artists", [])),
            }

        top_tracks_items = _retryable_call(self.sp.current_user_top_tracks, time_range=range, limit=min(50, limit)).get("items", [])
        top_artists_items = _retryable_call(self.sp.current_user_top_artists, time_range=range, limit=min(50, limit)).get("items", [])

        tr_rows = [{
            "track_id": t.get("id"),
            "track_name": t.get("name"),
            "artist_ids": [a.get("id") for a in t.get("artists", [])],
            "artist_names": ", ".join(a.get("name") for a in t.get("artists", [])),
            "album": t.get("album", {}).get("name"),
        } for t in top_tracks_items]

        ar_rows = [{
            "artist_id": a.get("id"),
            "artist_name": a.get("name"),
            "genres": a.get("genres", []),
            "popularity": a.get("popularity"),
        } for a in top_artists_items]

        result = {"tracks": tr_rows, "artists": ar_rows}
        self.cache.set(self.user_id, "top_tracks_artists", params, result, TOPS_TTL_SECONDS)

        return {"tracks": pd.DataFrame(tr_rows), "artists": pd.DataFrame(ar_rows)}

    def get_audio_features(self, track_ids: List[str]) -> pd.DataFrame:
        track_ids = [t for t in track_ids if t]
        if not track_ids:
            return pd.DataFrame()
        params = {"track_ids": sorted(track_ids)}
        cached = self.cache.get(self.user_id, "audio_features", params)
        if cached is not None:
            return pd.DataFrame(cached)

        rows: List[Dict[str, Any]] = []
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i : i + 100]
            feats = _retryable_call(self.sp.audio_features, tracks=batch)
            for f in feats:
                if f:
                    f = dict(f)
                    f["track_id"] = f.pop("id", None)
                    rows.append(f)
        df = pd.DataFrame(rows)
        self.cache.set(self.user_id, "audio_features", params, df.to_dict(orient="records"), AUDIO_TTL_SECONDS)
        return df

    def search_candidates(
        self,
        seed_artists: Optional[List[str]] = None,
        seed_genres: Optional[List[str]] = None,
        audio_feature_targets: Optional[Dict[str, float]] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        seed_artists = (seed_artists or [])[:5]
        seed_genres = (seed_genres or [])[:5]
        params = {
            "seed_artists": seed_artists,
            "seed_genres": seed_genres,
            "targets": audio_feature_targets or {},
            "limit": int(limit),
        }
        cached = self.cache.get(self.user_id, "search_candidates", params)
        if cached is not None:
            return pd.DataFrame(cached)

        query_params: Dict[str, Any] = {
            "seed_artists": seed_artists or None,
            "seed_genres": seed_genres or None,
            "limit": min(100, limit),
        }
        if audio_feature_targets:
            query_params.update({f"target_{k}": float(v) for k, v in audio_feature_targets.items()})

        recs = _retryable_call(self.sp.recommendations, **query_params)
        rows: List[Dict[str, Any]] = []
        for t in recs.get("tracks", []):
            rows.append({
                "track_id": t.get("id"),
                "track_name": t.get("name"),
                "artist_ids": [a.get("id") for a in t.get("artists", [])],
                "artist_names": ", ".join(a.get("name") for a in t.get("artists", [])),
                "album": t.get("album", {}).get("name"),
            })
        df = pd.DataFrame(rows)
        self.cache.set(self.user_id, "search_candidates", params, df.to_dict(orient="records"), RECS_TTL_SECONDS)
        return df
