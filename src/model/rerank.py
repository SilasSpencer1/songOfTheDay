from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import norm

from .mood import mood_to_feature_targets, mood_soft_score


AUDIO_FEATURE_COLUMNS: List[str] = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "tempo",
]


@dataclass(frozen=True)
class Weights:
    novelty: float = 0.45
    fit: float = 0.30
    mood: float = 0.20
    diversity: float = 0.05


def _safe_standardize(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x = df[list(cols)].astype(float).copy()
    means = x.mean(axis=0).to_numpy()
    stds = x.std(axis=0, ddof=0).to_numpy()
    stds = np.where(stds == 0, 1.0, stds)
    x_std = (x - means) / stds
    return x_std, means, stds


def _cosine_sim_matrix(matrix_a: np.ndarray, vector_b: np.ndarray) -> np.ndarray:
    if matrix_a.size == 0:
        return np.zeros((0,), dtype=float)
    denom = (norm(matrix_a, axis=1) * (norm(vector_b) + 1e-9)) + 1e-9
    return (matrix_a @ vector_b) / denom


def build_user_profile_vector(
    recent_tracks_df: pd.DataFrame,
    recent_audio_features_df: pd.DataFrame,
    max_n: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Join to bring played_at with features, order by recency
    if recent_tracks_df is None or recent_tracks_df.empty or recent_audio_features_df is None or recent_audio_features_df.empty:
        # Return zero vector and default stats to avoid NaNs downstream
        return np.zeros(len(AUDIO_FEATURE_COLUMNS), dtype=float), np.zeros(len(AUDIO_FEATURE_COLUMNS)), np.ones(len(AUDIO_FEATURE_COLUMNS))

    # Ensure proper columns
    feats = recent_audio_features_df.copy()
    if "track_id" not in feats.columns and "id" in feats.columns:
        feats = feats.rename(columns={"id": "track_id"})

    recent = recent_tracks_df[["track_id", "played_at"]].dropna().copy()
    if "played_at" in recent.columns:
        # sort descending by played_at
        recent["_ts"] = pd.to_datetime(recent["played_at"], errors="coerce")
        recent = recent.dropna(subset=["_ts"]).sort_values("_ts", ascending=False)
    recent = recent.drop_duplicates("track_id").head(max_n)

    joined = recent.merge(feats, on="track_id", how="inner")
    if joined.empty:
        return np.zeros(len(AUDIO_FEATURE_COLUMNS), dtype=float), np.zeros(len(AUDIO_FEATURE_COLUMNS)), np.ones(len(AUDIO_FEATURE_COLUMNS))

    x_std, means, stds = _safe_standardize(joined, AUDIO_FEATURE_COLUMNS)

    # Recency weights: exponential decay by rank (half-life ~ 50 tracks)
    ranks = np.arange(1, len(joined) + 1, dtype=float)
    half_life = 50.0
    weights = np.exp(-np.log(2.0) * (ranks - 1) / half_life)
    weights = weights / (weights.sum() + 1e-9)

    profile = (weights[:, None] * x_std.to_numpy()).sum(axis=0)
    return profile.astype(float), means.astype(float), stds.astype(float)


def rerank_candidates(
    candidates_df: pd.DataFrame,
    recent_tracks_df: pd.DataFrame,
    recent_audio_features_df: pd.DataFrame,
    mood: Optional[str] = None,
    weights: Weights = Weights(),
    exclude_explicit: bool = False,
    top_k: int = 20,
) -> List[Dict]:
    if candidates_df is None or candidates_df.empty:
        return []

    df = candidates_df.copy()

    # Optional: filter explicit
    if exclude_explicit and "explicit" in df.columns:
        df = df[~df["explicit"].astype(bool)]

    # Ensure required columns
    for col in ["track_id", "artist_id", "novelty"]:
        if col not in df.columns:
            df[col] = None

    # Build user profile and standardization stats on recent
    user_vec, means, stds = build_user_profile_vector(recent_tracks_df, recent_audio_features_df, max_n=200)

    # Compute fit score for candidates using same standardization
    x_cand = df[[c for c in AUDIO_FEATURE_COLUMNS if c in df.columns]].astype(float).copy()
    # Align columns fully (fill any missing with column mean over candidates before standardization)
    for c in AUDIO_FEATURE_COLUMNS:
        if c not in x_cand.columns:
            x_cand[c] = float(means[AUDIO_FEATURE_COLUMNS.index(c)]) if np.isfinite(means[AUDIO_FEATURE_COLUMNS.index(c)]) else 0.0
    x_cand = x_cand[AUDIO_FEATURE_COLUMNS]

    x_cand_std = (x_cand.to_numpy() - means) / (stds + 1e-9)
    fit_scores = _cosine_sim_matrix(x_cand_std, user_vec)

    # Mood targets and score
    target_ranges = mood_to_feature_targets(mood) if mood else None
    mood_scores = np.ones(len(df), dtype=float)
    if target_ranges:
        ms = []
        for _, row in df.iterrows():
            feat_map = {k: float(row[k]) for k in target_ranges.keys() if k in df.columns and pd.notna(row[k])}
            ms.append(mood_soft_score(feat_map, target_ranges))
        mood_scores = np.array(ms, dtype=float)

    # Novelty from precomputed column (0..1), fallback to 0.5
    nov = df["novelty"].astype(float).fillna(0.5).to_numpy()

    # Base score before diversity
    base = (
        weights.novelty * nov
        + weights.fit * fit_scores
        + weights.mood * mood_scores
    )

    # Greedy selection with diversity and constraint: max 1 per artist in top-10
    selected_indices: List[int] = []
    artist_counts: Dict[str, int] = {}
    genre_counts: Dict[str, int] = {}

    # Extract primary genre for each row
    def primary_genre(genres: object) -> Optional[str]:
        if isinstance(genres, list) and genres:
            return str(genres[0])
        return None

    genres_series = df["genres"].apply(primary_genre) if "genres" in df.columns else pd.Series([None] * len(df))

    remaining = list(range(len(df)))

    while remaining and len(selected_indices) < top_k:
        best_idx = None
        best_score = -1.0
        for idx in remaining:
            artist_id = str(df.iloc[idx]["artist_id"]) if pd.notna(df.iloc[idx]["artist_id"]) else None
            genre = genres_series.iloc[idx]

            # Diversity bonus in [0,1]: penalize repeats of artist/genre already selected
            a_count = artist_counts.get(artist_id or "", 0)
            g_count = genre_counts.get(genre or "", 0)
            diversity_bonus = max(0.0, 1.0 - 0.5 * min(1, a_count) - 0.5 * (g_count / 3.0))

            # Apply hard constraint for top-10: skip if artist already selected and we are still filling top-10
            if len(selected_indices) < 10 and a_count >= 1 and artist_id:
                continue

            score = base[idx] + weights.diversity * diversity_bonus
            if score > best_score:
                best_score = float(score)
                best_idx = idx
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        # Update diversity trackers
        artist_id = str(df.iloc[best_idx]["artist_id"]) if pd.notna(df.iloc[best_idx]["artist_id"]) else None
        genre = genres_series.iloc[best_idx]
        if artist_id:
            artist_counts[artist_id] = artist_counts.get(artist_id, 0) + 1
        if genre:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        remaining.remove(best_idx)

    # Build output with rationale
    results: List[Dict] = []
    for rank, idx in enumerate(selected_indices, start=1):
        row = df.iloc[idx]
        artist_id = row.get("artist_id")
        genre = genres_series.iloc[idx]
        a_count = artist_counts.get(str(artist_id) if artist_id is not None else "", 0)
        g_count = genre_counts.get(genre or "", 0)

        # Recompute diversity bonus for rationale
        diversity_bonus = max(0.0, 1.0 - 0.5 * min(1, a_count - 1) - 0.5 * ((g_count - 1) / 3.0))

        rationale_bits = []
        n = nov[idx]
        f = fit_scores[idx]
        m = mood_scores[idx]
        if n >= 0.9:
            rationale_bits.append("high novelty")
        elif n >= 0.6:
            rationale_bits.append("good novelty")
        else:
            rationale_bits.append("lower novelty")
        if f >= 0.6:
            rationale_bits.append("strong user fit")
        elif f >= 0.3:
            rationale_bits.append("moderate fit")
        if target_ranges:
            if m >= 0.7:
                rationale_bits.append("mood-aligned")
            elif m >= 0.5:
                rationale_bits.append("partial mood match")
        if diversity_bonus >= 0.9:
            rationale_bits.append("diverse pick")

        results.append({
            "track_id": row.get("track_id"),
            "artist_id": row.get("artist_id"),
            "name": row.get("name") or row.get("track_name"),
            "artist_name": row.get("artist_name") or row.get("artist_names"),
            "genres": row.get("genres"),
            "novelty": float(n),
            "fit_score": float(f),
            "mood_score": float(m),
            "final_score": float(base[idx] + weights.diversity * diversity_bonus),
            "rationale": ", ".join(rationale_bits),
            "rank": rank,
        })

    # Sort by final_score desc, then by novelty
    results.sort(key=lambda r: (r["final_score"], r["novelty"]), reverse=True)
    return results
