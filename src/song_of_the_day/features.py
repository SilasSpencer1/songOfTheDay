from __future__ import annotations

import numpy as np
import pandas as pd

AUDIO_FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "tempo",
]


def build_user_profile(audio_features_df: pd.DataFrame) -> np.ndarray:
    if audio_features_df.empty:
        return np.zeros(len(AUDIO_FEATURE_COLUMNS), dtype=float)
    feats = audio_features_df[AUDIO_FEATURE_COLUMNS].astype(float)
    feats = (feats - feats.mean()) / (feats.std(ddof=0) + 1e-9)
    centroid = feats.mean(axis=0).to_numpy()
    return centroid


def build_candidate_feature_matrix(cand_audio_df: pd.DataFrame) -> np.ndarray:
    if cand_audio_df.empty:
        return np.zeros((0, len(AUDIO_FEATURE_COLUMNS)), dtype=float)
    feats = cand_audio_df[AUDIO_FEATURE_COLUMNS].astype(float)
    feats = (feats - feats.mean()) / (feats.std(ddof=0) + 1e-9)
    return feats.to_numpy()
