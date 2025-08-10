from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from .config import Settings


SPOTIFY_SCOPES = "user-read-recently-played user-top-read"


class SpotifyService:
    def __init__(self, settings: Settings):
        self._sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=settings.spotify_client_id,
                client_secret=settings.spotify_client_secret,
                redirect_uri=settings.spotify_redirect_uri,
                scope=SPOTIFY_SCOPES,
                username=settings.spotify_username,
                open_browser=True,
                cache_path=None,
            )
        )

    def get_recent_tracks(self, limit: int = 50) -> pd.DataFrame:
        items = self._sp.current_user_recently_played(limit=limit)["items"]
        rows = []
        for item in items:
            track = item["track"]
            played_at = item["played_at"]
            rows.append(
                {
                    "track_id": track["id"],
                    "track_name": track["name"],
                    "artist_ids": [a["id"] for a in track["artists"]],
                    "artist_names": ", ".join(a["name"] for a in track["artists"]),
                    "album": track.get("album", {}).get("name"),
                    "played_at": played_at,
                }
            )
        return pd.DataFrame(rows)

    def get_top_artists(self, limit: int = 20, time_range: str = "medium_term") -> pd.DataFrame:
        items = self._sp.current_user_top_artists(limit=limit, time_range=time_range)["items"]
        rows = []
        for a in items:
            rows.append(
                {
                    "artist_id": a["id"],
                    "artist_name": a["name"],
                    "genres": a.get("genres", []),
                    "popularity": a.get("popularity"),
                }
            )
        return pd.DataFrame(rows)

    def get_artist_genres(self, artist_ids: Iterable[str]) -> Dict[str, List[str]]:
        artist_ids_list = list(dict.fromkeys([i for i in artist_ids if i]))
        genres_map: Dict[str, List[str]] = {}
        for i in range(0, len(artist_ids_list), 50):
            batch = artist_ids_list[i : i + 50]
            if not batch:
                continue
            artists = self._sp.artists(batch)["artists"]
            for a in artists:
                genres_map[a["id"]] = a.get("genres", [])
        return genres_map

    def get_audio_features(self, track_ids: Iterable[str]) -> pd.DataFrame:
        track_ids_list = list(dict.fromkeys([t for t in track_ids if t]))
        rows: List[Dict] = []
        for i in range(0, len(track_ids_list), 100):
            batch = track_ids_list[i : i + 100]
            if not batch:
                continue
            feats = self._sp.audio_features(batch)
            for f in feats:
                if f is None:
                    continue
                rows.append(f)
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.rename(columns={"id": "track_id"})
        return df

    def get_recommendations(
        self,
        seed_artists: Optional[List[str]] = None,
        seed_genres: Optional[List[str]] = None,
        seed_tracks: Optional[List[str]] = None,
        limit: int = 50,
        target_features: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        params: Dict[str, object] = {
            "seed_artists": seed_artists[:5] if seed_artists else None,
            "seed_genres": seed_genres[:5] if seed_genres else None,
            "seed_tracks": seed_tracks[:5] if seed_tracks else None,
            "limit": limit,
        }
        if target_features:
            params.update({f"target_{k}": float(v) for k, v in target_features.items()})
        recs = self._sp.recommendations(**params)
        rows = []
        for t in recs.get("tracks", []):
            rows.append(
                {
                    "track_id": t["id"],
                    "track_name": t["name"],
                    "artist_ids": [a["id"] for a in t["artists"]],
                    "artist_names": ", ".join(a["name"] for a in t["artists"]),
                    "album": t.get("album", {}).get("name"),
                }
            )
        return pd.DataFrame(rows)
