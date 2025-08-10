from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import norm

from .features import AUDIO_FEATURE_COLUMNS, build_candidate_feature_matrix


def cosine_similarity_matrix(matrix_a: np.ndarray, vector_b: np.ndarray) -> np.ndarray:
    if matrix_a.size == 0:
        return np.zeros((0,), dtype=float)
    denom = (norm(matrix_a, axis=1) * (norm(vector_b) + 1e-9)) + 1e-9
    sim = (matrix_a @ vector_b) / denom
    return sim


def rank_candidates(
    candidates_df: pd.DataFrame,
    cand_audio_df: pd.DataFrame,
    user_profile_vec: np.ndarray,
    novelty_scores: Dict[str, float],
    mood_targets: Optional[Dict[str, float]],
    weights: Dict[str, float],
    k_select: int = 20,
    diversity_lambda: float = 0.5,
) -> List[Tuple[str, float]]:
    if candidates_df.empty:
        return []

    cand_audio_df = cand_audio_df.set_index("track_id")
    feat_matrix = build_candidate_feature_matrix(cand_audio_df)

    user_fit = cosine_similarity_matrix(feat_matrix, user_profile_vec)

    mood_score = np.zeros(len(candidates_df), dtype=float)
    if mood_targets:
        for feat_name, target in mood_targets.items():
            if feat_name not in AUDIO_FEATURE_COLUMNS and feat_name != "tempo":
                continue
            x = cand_audio_df[feat_name].astype(float).to_numpy()
            x_norm = (x - x.mean()) / (x.std(ddof=0) + 1e-9)
            t_norm = (target - x.mean()) / (x.std(ddof=0) + 1e-9)
            mood_score += 1.0 - np.minimum(1.0, np.abs(x_norm - t_norm))
        mood_score /= max(1, len(mood_targets))

    novelty = np.array([novelty_scores.get(tid, 1.0) for tid in candidates_df["track_id"].tolist()])

    w_nov = float(weights.get("novelty", 0.35))
    w_fit = float(weights.get("user_fit", 0.35))
    w_mood = float(weights.get("mood", 0.20))

    base_score = w_nov * novelty + w_fit * user_fit + w_mood * mood_score

    selected: List[int] = []
    remaining: List[int] = list(range(len(candidates_df)))

    if feat_matrix.size == 0:
        pairwise = np.zeros((0, 0))
    else:
        denom = (norm(feat_matrix, axis=1, keepdims=True) @ norm(feat_matrix, axis=1, keepdims=True).T) + 1e-9
        pairwise = (feat_matrix @ feat_matrix.T) / denom

    while remaining and len(selected) < k_select:
        mmr_scores: List[Tuple[int, float]] = []
        for idx in remaining:
            if not selected:
                div_penalty = 0.0
            else:
                div_penalty = max(pairwise[idx, sel] for sel in selected)
            score = (1 - diversity_lambda) * base_score[idx] - diversity_lambda * div_penalty
            mmr_scores.append((idx, float(score)))
        mmr_scores.sort(key=lambda x: x[1], reverse=True)
        best_idx = mmr_scores[0][0]
        selected.append(best_idx)
        remaining.remove(best_idx)

    ranked = [(candidates_df.iloc[i]["track_id"], float(base_score[i])) for i in selected]
    return ranked
