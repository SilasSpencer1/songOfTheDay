from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import os
import pandas as pd

from .config import Settings, load_settings
from .spotify_client import SpotifyService
from .cache import CacheRepository
from .features import build_user_profile
from .novelty import compute_novelty_scores
from .mood import parse_mood_input
from .ranking import rank_candidates


@dataclass
class Recommendation:
    track_id: str
    track_name: str
    artist_names: str
    rationale: str


def generate_recommendations(settings: Optional[Settings] = None, mood_text: Optional[str] = None, rec_limit: int = 20) -> Dict:
    settings = settings or load_settings()

    use_data_client = os.getenv("SOTD_USE_DATA_CLIENT", "false").lower() in {"1", "true", "yes"}

    cache = CacheRepository(settings.db_path)
    cache.ensure_schema()

    if use_data_client:
        # Optional new data client path (incremental adoption)
        from data.spotify_client import SpotifyDataClient  # type: ignore

        client = SpotifyDataClient.create()

        recent_df = client.get_recent_tracks(limit=50, days_back=14)
        cache.record_listens(recent_df)

        tops = client.get_top_tracks_and_artists(range="medium_term", limit=20)
        top_artists_df = tops.get("artists", pd.DataFrame())
        seed_artists = top_artists_df["artist_id"].dropna().head(5).tolist() if not top_artists_df.empty else []
        seed_genres = sorted(list({g for gs in top_artists_df.get("genres", []) for g in (gs if isinstance(gs, list) else [])})) if not top_artists_df.empty else []

        mood_targets = parse_mood_input(mood_text)

        recs_df = client.search_candidates(
            seed_artists=seed_artists or None,
            seed_genres=seed_genres[:5] or None,
            audio_feature_targets=mood_targets,
            limit=max(50, rec_limit * 3),
        ).drop_duplicates("track_id")

        if recs_df.empty:
            return {"song_of_the_day": None, "alternates": [], "reason": "No candidates returned from Spotify."}

        cand_audio_df = client.get_audio_features(recs_df["track_id"].tolist())

        recent_audio_df = pd.DataFrame()
        if not recent_df.empty:
            recent_audio_df = client.get_audio_features(recent_df["track_id"].tolist())

    else:
        # Existing client path (default)
        spotify = SpotifyService(settings)

        recent_df = spotify.get_recent_tracks(limit=50)
        cache.record_listens(recent_df)

        top_artists_df = spotify.get_top_artists(limit=20, time_range="medium_term")
        seed_artists = top_artists_df["artist_id"].dropna().head(5).tolist()
        seed_genres = sorted(list({g for gs in top_artists_df.get("genres", []) for g in (gs if isinstance(gs, list) else [])}))

        mood_targets = parse_mood_input(mood_text)

        recs_df = spotify.get_recommendations(
            seed_artists=seed_artists or None,
            seed_genres=(seed_genres[:5] or None) if seed_genres else None,
            seed_tracks=None,
            limit=max(50, rec_limit * 3),
            target_features=mood_targets,
        ).drop_duplicates("track_id")

        if recs_df.empty:
            return {"song_of_the_day": None, "alternates": [], "reason": "No candidates returned from Spotify."}

        cand_audio_df = spotify.get_audio_features(recs_df["track_id"].tolist())

        recent_audio_df = pd.DataFrame()
        if not recent_df.empty:
            recent_audio_df = spotify.get_audio_features(recent_df["track_id"].tolist())

    user_profile_vec = build_user_profile(recent_audio_df)

    recent_hist_df = cache.get_recent_history(days=14)
    lifetime_counts = cache.get_lifetime_counts()
    novelty_scores = compute_novelty_scores(
        candidate_ids=recs_df["track_id"].tolist(),
        recent_history=recent_hist_df,
        lifetime_counts=lifetime_counts,
        now_utc=datetime.utcnow(),
    )

    weights = {
        "novelty": settings.weight_novelty,
        "user_fit": settings.weight_user_fit,
        "mood": settings.weight_mood,
    }

    ranked = rank_candidates(
        candidates_df=recs_df,
        cand_audio_df=cand_audio_df,
        user_profile_vec=user_profile_vec,
        novelty_scores=novelty_scores,
        mood_targets=mood_targets,
        weights=weights,
        k_select=max(6, rec_limit),
        diversity_lambda=settings.weight_diversity,
    )

    if not ranked:
        return {"song_of_the_day": None, "alternates": [], "reason": "Ranking produced no results."}

    id_to_row = recs_df.set_index("track_id").to_dict(orient="index")
    recs: List[Recommendation] = []
    for track_id, score in ranked:
        row = id_to_row.get(track_id, {})
        rationale_bits = []
        nov = novelty_scores.get(track_id, 0.0)
        if nov >= 0.9:
            rationale_bits.append("high novelty")
        elif nov >= 0.6:
            rationale_bits.append("good novelty")
        else:
            rationale_bits.append("lower novelty")
        if mood_targets:
            rationale_bits.append(f"matches mood '{mood_text}'")
        recs.append(
            Recommendation(
                track_id=track_id,
                track_name=row.get("track_name", track_id),
                artist_names=row.get("artist_names", ""),
                rationale=", ".join(rationale_bits),
            )
        )

    song_of_the_day = recs[0]
    alternates = recs[1:6]

    return {
        "song_of_the_day": song_of_the_day.__dict__,
        "alternates": [r.__dict__ for r in alternates],
    }
