from __future__ import annotations

from typing import Dict, Tuple, Optional

# Canonical feature bounds used for normalization
FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "valence": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "danceability": (0.0, 1.0),
    "acousticness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "tempo": (50.0, 200.0),
    "loudness": (-30.0, 0.0),
}


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _range_around(target: float, half_width: float, bounds: Tuple[float, float]) -> Tuple[float, float]:
    lo = _clip(target - half_width, bounds[0], bounds[1])
    hi = _clip(target + half_width, bounds[0], bounds[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


# Mood lexicon with valence–arousal anchors and additional feature heuristics
# Values are midpoints; we expand to ranges with sensible half-widths per mood
_MOOD_ANCHORS: Dict[str, Dict[str, float]] = {
    "happy": {"valence": 0.9, "energy": 0.7, "danceability": 0.75, "tempo": 120, "loudness": -6},
    "calm": {"valence": 0.6, "energy": 0.25, "acousticness": 0.6, "tempo": 80, "loudness": -14},
    "melancholic": {"valence": 0.2, "energy": 0.35, "acousticness": 0.6, "tempo": 70, "loudness": -16},
    "angry": {"valence": 0.2, "energy": 0.85, "danceability": 0.55, "tempo": 140, "loudness": -4},
    "romantic": {"valence": 0.7, "energy": 0.45, "acousticness": 0.5, "danceability": 0.6, "tempo": 95, "loudness": -10},
    "focused": {"valence": 0.5, "energy": 0.3, "instrumentalness": 0.7, "tempo": 90, "loudness": -12},
    "hype": {"valence": 0.8, "energy": 0.9, "danceability": 0.85, "tempo": 130, "loudness": -5},
    "chill": {"valence": 0.6, "energy": 0.35, "danceability": 0.65, "tempo": 85, "loudness": -12},
}

# Per-feature default half-widths controlling range tightness around anchors
_DEFAULT_HALF_WIDTH: Dict[str, float] = {
    "valence": 0.15,
    "energy": 0.2,
    "danceability": 0.2,
    "acousticness": 0.25,
    "instrumentalness": 0.3,
    "tempo": 15.0,
    "loudness": 4.0,
}

# Mood-specific looseness multipliers
_MOOD_SPREAD: Dict[str, float] = {
    "happy": 1.0,
    "calm": 1.2,
    "melancholic": 1.1,
    "angry": 0.9,
    "romantic": 1.0,
    "focused": 1.1,
    "hype": 0.9,
    "chill": 1.2,
}


def default_target_ranges() -> Dict[str, Tuple[float, float]]:
    # Wide defaults (no mood)
    return {
        "valence": FEATURE_BOUNDS["valence"],
        "energy": FEATURE_BOUNDS["energy"],
        "danceability": FEATURE_BOUNDS["danceability"],
        "acousticness": FEATURE_BOUNDS["acousticness"],
        "instrumentalness": FEATURE_BOUNDS["instrumentalness"],
        "tempo": FEATURE_BOUNDS["tempo"],
        "loudness": FEATURE_BOUNDS["loudness"],
    }


def mood_to_feature_targets(mood: Optional[str]) -> Dict[str, Tuple[float, float]]:
    if not mood:
        return default_target_ranges()
    key = mood.strip().lower()
    anchors = _MOOD_ANCHORS.get(key)
    if not anchors:
        return default_target_ranges()

    spread = _MOOD_SPREAD.get(key, 1.0)
    ranges: Dict[str, Tuple[float, float]] = {}
    for feat, bounds in FEATURE_BOUNDS.items():
        if feat in anchors:
            half = _DEFAULT_HALF_WIDTH[feat] * spread
            ranges[feat] = _range_around(anchors[feat], half, bounds)
        else:
            # If not anchored, provide a slightly relaxed default range
            lo, hi = FEATURE_BOUNDS[feat]
            mid = (lo + hi) / 2
            half = _DEFAULT_HALF_WIDTH[feat] * spread
            ranges[feat] = _range_around(mid, half, bounds)

    return ranges


def mood_soft_score(audio_features: Dict[str, float], target_ranges: Dict[str, Tuple[float, float]]) -> float:
    # Normalized average distance outside ranges, then convert to score in [0,1]
    if not audio_features or not target_ranges:
        return 1.0

    distances = []
    for feat, (min_t, max_t) in target_ranges.items():
        if feat not in audio_features:
            continue
        val = float(audio_features[feat])
        # Distance to range: 0 if inside, else positive gap to nearest bound
        if min_t <= val <= max_t:
            d = 0.0
        else:
            d = min(abs(val - min_t), abs(val - max_t))
        # Normalize by global feature span to keep distances comparable across features
        gmin, gmax = FEATURE_BOUNDS.get(feat, (min_t, max_t))
        span = max(gmax - gmin, 1e-9)
        distances.append(d / span)

    if not distances:
        return 1.0

    avg_dist = sum(distances) / len(distances)
    score = 1.0 - max(0.0, min(1.0, avg_dist))
    return float(score)
