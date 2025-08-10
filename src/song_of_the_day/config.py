from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    spotify_client_id: str
    spotify_client_secret: str
    spotify_redirect_uri: str
    spotify_username: Optional[str]
    db_path: Path
    weight_novelty: float = 0.35
    weight_user_fit: float = 0.35
    weight_mood: float = 0.20
    weight_diversity: float = 0.10


def load_settings() -> Settings:
    load_dotenv()

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8080/callback")
    username = os.getenv("SPOTIFY_USERNAME")

    if not client_id or not client_secret:
        raise RuntimeError("Missing Spotify credentials. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env.")

    db_path_str = os.getenv("SOTD_DB_PATH", "./data/sotd_cache.db")
    db_path = Path(db_path_str).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    weight_novelty = float(os.getenv("SOTD_WEIGHT_NOVELTY", 0.35))
    weight_user_fit = float(os.getenv("SOTD_WEIGHT_USER_FIT", 0.35))
    weight_mood = float(os.getenv("SOTD_WEIGHT_MOOD", 0.20))
    weight_diversity = float(os.getenv("SOTD_WEIGHT_DIVERSITY", 0.10))

    return Settings(
        spotify_client_id=client_id,
        spotify_client_secret=client_secret,
        spotify_redirect_uri=redirect_uri,
        spotify_username=username,
        db_path=db_path,
        weight_novelty=weight_novelty,
        weight_user_fit=weight_user_fit,
        weight_mood=weight_mood,
        weight_diversity=weight_diversity,
    )
