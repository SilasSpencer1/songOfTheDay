from __future__ import annotations

from typing import Dict, Optional


DEFAULT_MOODS: Dict[str, Dict[str, float]] = {
    "happy": {"valence": 0.9, "energy": 0.7, "danceability": 0.7},
    "sad": {"valence": 0.2, "energy": 0.3, "acousticness": 0.6},
    "chill": {"valence": 0.6, "energy": 0.3, "danceability": 0.6},
    "focus": {"valence": 0.5, "energy": 0.25, "instrumentalness": 0.7},
    "party": {"valence": 0.8, "energy": 0.9, "danceability": 0.9},
    "angry": {"valence": 0.2, "energy": 0.85},
}


def parse_mood_input(mood_text: Optional[str]) -> Optional[Dict[str, float]]:
    if not mood_text:
        return None
    key = mood_text.strip().lower()
    return DEFAULT_MOODS.get(key)
